# Brain Tumor Segmentation Using Deep Learning on the BraTS2020 Dataset

> Neural Networks Course Project
> Authors: Stanisław Burdzicki, Kacper Faliński

---

# 1. Introduction

Brain tumor segmentation is a fundamental task in medical image analysis. Accurate delineation of tumor regions can support diagnosis, treatment planning, and monitoring of disease progression. Manual segmentation performed by radiologists is time-consuming and subject to inter-observer variability, making automated approaches highly desirable.

The objective of this project was to develop and evaluate deep learning models capable of performing semantic segmentation of brain tumors in MRI scans from the BraTS2020 dataset. The project focused on U-Net-based architectures and explored various preprocessing techniques, loss functions, and training strategies to improve segmentation quality while maintaining computational efficiency.

---

# 2. Dataset

The project utilizes the BraTS2020 (Brain Tumor Segmentation Challenge 2020) training dataset.

## 2.1 Dataset Characteristics

The dataset contains:

* 369 MRI volumes
* 155 slices per volume
* Resolution of 240 × 240 pixels
* Four MRI modalities:

  * T1
  * T1ce
  * T2
  * FLAIR
* Expert-annotated segmentation masks

Each voxel belongs to one of the following classes:

| Label | Class                               |
| ----- | ----------------------------------- |
| 0     | Background                          |
| 1     | Necrotic / Non-Enhancing Tumor Core |
| 2     | Peritumoral Edema                   |
| 4     | Enhancing Tumor                     |

For training purposes, label 4 was remapped to label 3 to create a continuous class numbering scheme.

---

# 3. Data Exploration

Before model development, an exploratory analysis of the dataset was performed.

## 3.1 NIfTI Format

The MRI scans are stored in NIfTI (`.nii`) format, which contains both volumetric image data and spatial metadata. Data loading was performed using the `nibabel` library.

## 3.2 Dataset Consistency

All MRI volumes contain exactly 155 slices, indicating a complete dataset without missing scans.

## 3.3 Class Distribution

The dataset exhibits severe class imbalance. Background voxels dominate the dataset and account for the vast majority of all pixels, while tumor regions occupy only a small fraction of each scan.

This imbalance motivated the use of weighted loss functions during training.

## 3.4 Empty Slices

Many slices at the beginning and end of each volume contain only empty space surrounding the brain. Including these slices in training would provide little useful information while increasing computational cost.

## 3.5 Brain Bounding Box

A global brain bounding box was computed across all training volumes:

```text
x: [41, 195]
y: [29, 222]
```

This information was later incorporated into the preprocessing pipeline to remove unnecessary background regions.

---

# 4. Project Development

The project evolved through several stages, each addressing limitations identified in previous experiments.

## 4.1 Stage 1 – Initial Data Loading and Lazy Loading

The first implementation relied on loading MRI volumes directly from NIfTI files during training.

Due to the large size of the dataset, storing multiple volumes simultaneously resulted in excessive memory usage. To reduce memory consumption, a lazy loading mechanism was implemented that loaded only the data required for the current batch.

While this approach solved the memory issue, it introduced a significant performance bottleneck. Frequent disk access and repeated NIfTI parsing caused the GPU to remain idle while waiting for data preparation.

---

## 4.2 Stage 2 – Loss Function Experiments

In parallel with data-loading optimizations, several loss functions were evaluated.

### CrossEntropyLoss

Standard CrossEntropyLoss served as the baseline approach. However, because of the severe class imbalance, the model tended to prioritize background prediction.

### Weighted CrossEntropyLoss

To compensate for class imbalance, higher weights were assigned to tumor classes:

```python
weights = [0.1, 2.0, 2.0, 2.0]
```

This significantly improved tumor segmentation performance while maintaining low computational overhead.

### Dice Loss

Dice Loss was evaluated using the MONAI framework. Although Dice Loss is widely used in medical image segmentation, training became noticeably slower compared to CrossEntropyLoss.

### DiceCELoss

A combination of Dice Loss and CrossEntropy Loss was also tested. While segmentation quality remained competitive, training time increased without providing substantial improvements over weighted CrossEntropyLoss.

### Final Choice

Weighted CrossEntropyLoss was selected as the final loss function due to its favorable balance between computational efficiency and segmentation quality.

---

## 4.3 Stage 3 – Offline Preprocessing and Dataset Caching

To eliminate the CPU bottleneck identified in Stage 1, an offline preprocessing pipeline was introduced.

The preprocessing workflow consisted of:

1. Loading NIfTI volumes
2. Per-modality normalization
3. Resizing slices to 128 × 128 pixels
4. Saving processed volumes as `.pt` files

By storing preprocessed tensors directly, all training data could be loaded into RAM before training began.

At this stage, two MRI modalities were selected as model inputs:

* FLAIR
* T1ce

These modalities provided complementary information regarding tumor boundaries and active tumor regions.

Additionally, slices containing only background were randomly discarded with an 80% probability, improving class balance and reducing training time.

---

## 4.4 Stage 4 – Cloud Training and Improved Cropping

The next objective was to introduce Batch Normalization and increase batch size.

Initial benchmarks indicated that local training would require more than twelve hours per experiment. Consequently, training was migrated to cloud infrastructure.

The project was containerized using Docker and executed on rented GPUs through RunPod.

At the same time, preprocessing was refined using the previously computed brain bounding box.

The updated preprocessing pipeline included:

1. Cropping each slice to the brain region
2. Padding the cropped image to a square shape
3. Resizing to 128 × 128 pixels

This reduced the amount of irrelevant background while preserving anatomical structures.

Final cloud-training configuration:

* 369 MRI volumes
* 40 epochs
* Batch size of 6

Additionally, class weight tuning based on class frequency was introduced during training. Instead of manually selected weights, class weights were computed as the square root of the inverse class frequency in the training dataset. This approach was used to better reflect the natural distribution of classes and further mitigate the effects of class imbalance.

---

## 4.5 Stage 5 – Final Models

Three architectures were compared.

### UNet

A standard U-Net implementation used as the baseline model.

### UNetNorm

An enhanced U-Net architecture with Batch Normalization layers added after each convolution.

The goal was to improve optimization stability and accelerate convergence.

### UNetResNet

A U-Net architecture employing a pretrained ResNet-34 encoder.

Training followed a progressive unfreezing strategy:

1. Decoder-only training
2. Partial encoder unfreezing
3. Full encoder fine-tuning with differential learning rates

This approach leveraged ImageNet pretraining while minimizing catastrophic forgetting.

---

# 5. Model Architectures

All models use two-channel inputs (FLAIR and T1ce) and produce four-class segmentation masks.

## 5.1 UNet / UNetNorm

The architecture consists of:

* Four encoder stages
* Four decoder stages
* Skip connections between matching resolutions
* Final pixel-wise classification layer

UNetNorm extends this design by incorporating Batch Normalization after each convolutional layer.

## 5.2 UNetResNet

The encoder is replaced by a pretrained ResNet-34 backbone, while the decoder follows the standard U-Net design.

This architecture aims to benefit from features learned on large-scale image datasets.

---

# 6. Results

Performance was evaluated using the Dice coefficient on the validation set (approximately 10% of the dataset).

| Model                        | Necrotic Core | Enhancing Tumor |     Edema | Background |
| ---------------------------- | ------------: | --------------: | --------: | ---------: |
| UNet                         |         0.566 |           0.760 |     0.582 |      0.989 |
| UNet (Tuned Weights)         |         0.538 |           0.752 |     0.652 |      0.992 |
| UNetNorm                     |         0.604 |           0.817 |     0.674 |      0.993 |
| **UNetNorm (Tuned Weights)** |     **0.625** |       **0.818** | **0.721** |  **0.995** |
| UNetResNet (Tuned Weights)   |         0.595 |           0.791 |     0.726 |      0.995 |

---

# 7. Discussion

The experiments demonstrated that architectural improvements and preprocessing choices significantly influence segmentation quality.

The introduction of Batch Normalization consistently improved performance across all tumor classes. The largest improvements were observed for enhancing tumor and edema segmentation.

Class-weight tuning further improved segmentation of underrepresented classes, particularly edema.

The UNetResNet architecture achieved competitive performance and produced the highest Dice score for edema segmentation. This suggests that pretrained encoders are a promising direction for future work.

---

# 8. Conclusion

This project successfully developed and evaluated multiple deep learning approaches for brain tumor segmentation using the BraTS2020 dataset.

The best overall model was UNetNorm with tuned class weights, achieving the strongest balance across all tumor classes while maintaining efficient training.

Future improvements could include:

- 3D segmentation architectures
- Advanced data augmentation
- Additional hyperparameter optimization

These directions may further improve segmentation accuracy and model robustness in clinical applications.
