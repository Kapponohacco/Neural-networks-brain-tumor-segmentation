# Brain Tumor Segmentation with U-Net (BraTS2020)

> Neural Networks Course Project

## Overview

This project focuses on semantic segmentation of brain tumors in MRI scans using U-Net-based deep learning architectures trained on the BraTS2020 dataset.

The goal is to classify each pixel into one of four classes:

| Class | Description |
|---------|-------------|
| 0 | Background |
| 1 | Necrotic / Non-Enhancing Tumor Core (NCR) |
| 2 | Peritumoral Edema |
| 3 | Enhancing Tumor |

The project explores different model architectures, preprocessing strategies, and class-balancing techniques to improve segmentation performance.

## Dataset

The models were trained using the **BraTS2020 (Brain Tumor Segmentation Challenge 2020)** dataset.

### Dataset Statistics

- 369 MRI volumes
- 155 slices per volume
- Resolution: 240 × 240 pixels
- Four MRI modalities available:
  - T1
  - T1ce
  - T2
  - FLAIR

For this project, only **FLAIR** and **T1ce** modalities were used as model inputs.

## Preprocessing

The preprocessing pipeline consisted of:

1. Brain-region cropping using a dataset-wide bounding box
2. Per-modality intensity normalization
3. Padding cropped images to square dimensions
4. Resizing slices to 128 × 128
5. Offline caching to `.pt` files for faster training

Additionally, slices containing only background were downsampled to reduce class imbalance and accelerate training.

## Models

### UNet

Baseline U-Net implementation used as the primary reference model.

### UNetNorm

U-Net enhanced with Batch Normalization layers after each convolutional block.

### UNetResNet

U-Net architecture with a pretrained ResNet-34 encoder initialized from ImageNet weights. The encoder was trained using a gradual unfreezing strategy to improve transfer learning performance.

## Training

### Loss Function

The final models were trained using weighted Cross Entropy Loss.

Additionally, we experimented with class weight tuning based on the actual frequency of classes in the training dataset, where weights were computed as the square root of the inverse class frequency, instead of manually chosen weights [0.1, 2.0, 2.0, 2.0].

### Final Training Configuration

- Dataset size: 369 MRI volumes
- Epochs: 40
- Batch size: 6
- Input channels: 2 (FLAIR + T1ce)
- Output classes: 4

Training was performed on cloud GPUs using RunPod.

## Results

Dice score evaluated on the validation set (~10% of the dataset).

| Model | Necrotic Core | Enhancing Tumor | Edema | Background |
|---------|---------:|---------:|---------:|---------:|
| UNet | 0.566 | 0.760 | 0.582 | 0.989 |
| UNet (Tuned Weights) | 0.538 | 0.752 | 0.652 | 0.992 |
| UNetNorm | 0.604 | 0.817 | 0.674 | 0.993 |
| **UNetNorm (Tuned Weights)** | **0.625** | **0.818** | **0.721** | **0.995** |
| UNetResNet (Tuned Weights) | 0.595 | 0.791 | 0.726 | 0.995 |

### Key Findings

- Batch Normalization significantly improved segmentation quality across all tumor classes.
- Class-weight tuning improved edema segmentation performance.
- Brain-region cropping and offline caching greatly reduced training overhead.
- The best overall model was **UNetNorm with tuned class weights**.

## Technologies

- Python
- PyTorch
- MONAI
- NumPy
- OpenCV
- Nibabel
- Docker
- RunPod

## Authors

- Stanisław Burdzicki
- Kacper Faliński
