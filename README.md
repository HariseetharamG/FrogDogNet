# FrogDogNet - EarthVision CVPRw2025

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of the  paper **"FrogDogNet"**.

This repository contains the code for our static Fourier cutoff approach to visual prompt tuning, designed specifically to filter high-frequency sensor noise and spatial clutter in remote sensing imagery.

## 🚀 Main Contribution
* **Static Fourier Cutoff Mechanism:** We introduce a  frequency thresholding module in the Fourier domain. This prevents the Meta-Net from overfitting to domain-specific high-frequency spatial clutter, enabling superior out-of-distribution generalization compared to standard spatial-only adapters like CoCoOp.

## 🛠️ Environment Setup
Our framework is built on top of the [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) engine.

```bash
# Clone the repository
git clone [https://github.com/HariseetharamG/FrogDogNet.git](https://github.com/HariseetharamG/FrogDogNet.git)
cd FrogDogNet

# Create and activate conda environment
conda create -n frogdognet python=3.8
conda activate frogdognet

# Install PyTorch and dependencies (adjust CUDA version as needed)
pip install torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
pip install -r requirements.txt

