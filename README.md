# Weather Translation GAN Models  
This repository contains two deep learning models for **cross-domain weather translation**, converting weather-affected images into clear-weather images.

The included models are:

1. **Cloudy → Clear using cGAN**  
   File: `pcloudyto-clear-cgan.ipynb`

2. **Snowy → Clear using CycleGAN**  
   File: `Snowy_to_clear_cycleGAN.ipynb`

---

## 📌 Project Overview
Both models are trained on a custom dataset containing four weather conditions:
- Cloudy  
- Snowy  
- Clear  
- Lightning  

The goal is to translate weather-affected images into clear images.

---

## 🚀 Model Architectures

### **1. cGAN (Cloudy → Clear)**
- Generator: Encoder–Decoder (ResNet inspired)  
- Discriminator: PatchGAN  
- Losses:
  - Adversarial Loss  
  - L1 Loss  
- Optimizer: Adam

### **2. CycleGAN (Snowy → Clear)**
- Two Generators: G (Snowy→Clear), F (Clear→Snowy)  
- Two Discriminators: D_Snowy, D_Clear  
- Losses:
  - Adversarial Loss  
  - Cycle-Consistency Loss  
  - Identity Loss  
- Optimizer: Adam

---

## 📂 Dataset Description
Your dataset folder structure should be:

dataset/
cloudy/
clear/
snowy/
lightning/



Preprocessing used in notebooks:
- Resize images to 256×256  
- Normalize to [-1, 1]  
- Paired data for cGAN and unpaired data for CycleGAN  

---

## 📊 Evaluation Metric — FID Only
The only evaluation metric used is:

### **FID — Fréchet Inception Distance**
- Lower FID = Better realism  
- Computed using InceptionV3 features  

Both notebooks include:
- FID computation block  
- Sample image generation  
- Final FID scores  

---

## 🛠️ How to Run
### 1. Install required libraries
```bash
pip install tensorflow keras numpy pillow matplotlib scikit-image tqdm



├── pcloudyto-clear-cgan.ipynb
├── Snowy_to_clear_cycleGAN.ipynb
├── README.md
├── models/
│   ├── cgan_generator.h5
│   ├── cgan_discriminator.h5
│   ├── cyclegan_G.h5
│   └── cyclegan_F.h5
├── results/
│   ├── cloudy_to_clear/
│   └── snowy_to_clear/
└── dataset/
