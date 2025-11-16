import os, random, math, contextlib
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

rainy_dir = r"data\dataset\rainy"
clear_dir = r"data\dataset\clear"

save_model_dir = "./checkpoints_fastcut_rainy2clear_updated"
save_image_dir = "./results_fastcut_rainy2clear_updated"

os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_image_dir, exist_ok=True)

resume_checkpoint_epoch = 5
start_from_epoch = 1

epochs = 200
batch_size = 2
img_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 2e-4
lambda_NCE = 1.0
lambda_L1 = 10.0
n_res_blocks = 9
decay_start_epoch = epochs // 2

if device == "cuda":
    autocast_ctx = torch.cuda.amp.autocast
else:
    autocast_ctx = contextlib.nullcontext

class UnpairedDataset(Dataset):
    def __init__(self, a_dir, b_dir, transform):
        self.a = [os.path.join(a_dir, f) for f in os.listdir(a_dir)]
        self.b = [os.path.join(b_dir, f) for f in os.listdir(b_dir)]
        self.a = [x for x in self.a if x.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.b = [x for x in self.b if x.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(self.a)==0 or len(self.b)==0:
            raise RuntimeError("Empty dataset folder.")
        self.transform = transform

    def __len__(self):
        return max(len(self.a), len(self.b))

    def __getitem__(self, idx):
        ra = Image.open(self.a[idx % len(self.a)]).convert("RGB")
        cb = Image.open(self.b[random.randint(0, len(self.b)-1)]).convert("RGB")
        return self.transform(ra), self.transform(cb)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

dataset = UnpairedDataset(rainy_dir, clear_dir, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU()
        )
        self.middle = nn.Sequential(*[ResnetBlock(256) for _ in range(n_res_blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.InstanceNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x, return_features=False):
        feats = []
        out = x
        for l in self.encoder:
            out = l(out)
            if isinstance(l, nn.ReLU):
                feats.append(out)
        out = self.middle(out)
        if return_features:
            return self.decoder(out), feats
        return self.decoder(out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,512,4,1,1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2,True),
            nn.Conv2d(512,1,4,1,1)
        )
    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()

class PatchNCELoss(nn.Module):
    def __init__(self, temp=0.07, max_samples=1024):
        super().__init__()
        self.temp = temp
        self.max_samples = max_samples

    def forward(self, f_src, f_tgt):
        B, C, H, W = f_src.shape
        a = f_src.view(B, C, -1)
        b = f_tgt.view(B, C, -1)
        if a.shape[2] > self.max_samples:
            idx = torch.randperm(a.shape[2], device=a.device)[:self.max_samples]
            a = a[:, :, idx]
            b = b[:, :, idx]
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        logits = torch.bmm(a.permute(0,2,1), b) / self.temp
        labels = torch.arange(logits.size(1), device=a.device).unsqueeze(0).repeat(B,1)
        return F.cross_entropy(logits, labels)

NCE = PatchNCELoss()

ckpt_path = os.path.join(save_model_dir, f"G_epoch_{resume_checkpoint_epoch}.pth")

if os.path.isfile(ckpt_path):
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    start_from_epoch = resume_checkpoint_epoch + 1
    print(f"Loaded checkpoint: {ckpt_path}")
else:
    print("No checkpoint found. Starting fresh.")

scaler = torch.cuda.amp.GradScaler() if device=="cuda" else None
best_fid = 9999

for epoch in range(start_from_epoch, epochs+1):
    loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

    if epoch > decay_start_epoch:
        lr_new = lr * (1 - (epoch - decay_start_epoch) / (epochs - decay_start_epoch))
        for g in opt_G.param_groups: g['lr'] = lr_new
        for g in opt_D.param_groups: g['lr'] = lr_new

    for rainy, clear in loop:
        rainy = rainy.to(device)
        clear = clear.to(device)

        with autocast_ctx():
            fake_clear, _ = G(rainy, return_features=True)
            pred_real = D(clear)
            pred_fake = D(fake_clear.detach())
            loss_D = 0.5 * (
                criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            )

        opt_D.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
        else:
            loss_D.backward()
            opt_D.step()

        with autocast_ctx():
            fake_clear, feats_q = G(rainy, return_features=True)
            loss_G_GAN = criterion_GAN(D(fake_clear), torch.ones_like(D(fake_clear)))
            loss_L1 = criterion_L1(fake_clear, clear) * lambda_L1
            _, feats_k = G(clear, return_features=True)
            loss_NCE = 0
            for fq, fk in zip(feats_q, feats_k):
                loss_NCE += NCE(fq, fk)
            loss_NCE *= lambda_NCE
            loss_G = loss_G_GAN + loss_L1 + loss_NCE

        opt_G.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()
        else:
            loss_G.backward()
            opt_G.step()

        loop.set_postfix({"lossD": float(loss_D), "lossG": float(loss_G), "NCE": float(loss_NCE)})

    torch.save(G.state_dict(), os.path.join(save_model_dir, f"G_epoch_{epoch}.pth"))

    with torch.no_grad():
        rainy_sample = rainy[:1]
        fake_prev = G(rainy_sample) * 0.5 + 0.5
        utils.save_image(fake_prev, os.path.join(save_image_dir, f"epoch_{epoch}.png"))

print("Training finished successfully.")
