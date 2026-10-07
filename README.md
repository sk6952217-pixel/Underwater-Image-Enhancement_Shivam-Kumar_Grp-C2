# Underwater Image Enhancement Using Classical Methods and CNN

##  Project Overview

This project focuses on enhancing degraded underwater images using both classical image-processing techniques and a learning-based CNN approach.

Underwater images commonly suffer from:

- Low visibility
- Poor contrast
- Color distortion
- Loss of visual details
- Weak edge information

The main objective of this project is to improve underwater image quality and compare traditional enhancement methods with a lightweight CNN-based image-to-image enhancement model.

The complete workflow followed in this project is:

Teacher Dataset
→ Dataset Understanding
→ Quality Checking
→ Preprocessing
→ Train/Validation/Test Split
→ Classical Enhancement
→ Metric Evaluation
→ CNN Model
→ Training and Validation
→ Parameter Refinement
→ Final Evaluation
→ Inference Time
→ Failure Analysis
→ Final Report and Software Package


#  Dataset

The dataset contains paired images:

- Input: degraded underwater image
- Target: corresponding enhanced/reference image

Final dataset:

- Total pairs: 153
- Training pairs: 107
- Validation pairs: 23
- Test pairs: 23

Dataset split:

- Training: 70%
- Validation: 15%
- Testing: 15%

Image configuration:

- Image format: PNG
- Image size: 256 × 256
- Channels: 3
- Color format: RGB
- Normalization: 0–1
- Batch size: 8
- Data augmentation: None
- seed: 20260922

The dataset was loaded using PyTorch Dataset and DataLoader.


# Project Work: Day 1–10

## Day 1–10: Dataset Understanding and Initial Preparation

During the initial stage, the underwater dataset was downloaded and extracted in Google Colab.

The dataset structure was inspected and the input and target image folders were identified.

The main objective of this stage was to understand:

- Dataset organization
- Input images
- Target images
- Image formats
- Image dimensions
- Input-target pairing

A basic image-loading pipeline was created using Python and Pillow.

Images were converted to RGB and resized to:

256 × 256

Pixel values were normalized from:

0–255

to:

0–1

A sample input and target image was also checked to verify the preprocessing.


# Dataset Loading

A custom PyTorch Dataset class was created.

The Dataset:

1. Reads input images.
2. Reads the corresponding target images.
3. Converts both images to RGB.
4. Converts them into tensors.
5. Normalizes pixel values.
6. Returns the input-target pair.

DataLoaders were created for:

- Training
- Validation
- Testing

The final batch size used was 8.

A sample batch was checked and the tensor shape was:

[8, 3, 256, 256]

for both input and target images.


#  Dataset Configuration

A preprocessing configuration file was created to record the final dataset setup.

Important configuration:

- Version: Teacher_Dataset
- Image size: 256 × 256
- Format: PNG
- Channels: 3
- Normalization: 0–1
- Training pairs: 107
- Validation pairs: 23
- Test pairs: 23
- Total pairs: 153
- Batch size: 8
- Augmentation: None
- Split: 70% / 15% / 15%
- Seed: 20260922


# Day 11–12: Preprocessing and Dataset Verification

The preprocessing pipeline was finalized.

The main preprocessing operations were:

- RGB conversion
- Image resizing
- Pixel normalization
- Input-target pairing
- Dataset splitting
- DataLoader creation

The final dataset was checked before starting enhancement experiments.

This ensured that the same train, validation and test data could be used consistently for the remaining experiments.


#  Day 13: Classical Enhancement Baselines

Four classical enhancement methods were implemented.

##  Histogram Equalization

Histogram Equalization was applied to the luminance component of the image.

Purpose:

- Improve global contrast
- Make dark regions more visible

##  CLAHE

CLAHE stands for Contrast Limited Adaptive Histogram Equalization.

It was applied in LAB color space.

Purpose:

- Improve local contrast
- Enhance details
- Limit excessive contrast amplification

##  Gamma Correction

Gamma Correction was used to modify image brightness.

Different gamma values were later tested during refinement.

Purpose:

- Improve visibility
- Adjust brightness
- Recover details in dark underwater images

##  White Balance

White Balance was implemented using LAB color space.

Purpose:

- Reduce color cast
- Improve color balance
- Compensate for underwater color distortion

The enhanced images from all four methods were saved for evaluation and visual comparison.


# Classical Baseline Results

The initial classical methods were evaluated using PSNR and SSIM.

Edge preservation was also evaluated using Edge F1.

Final initial baseline results:

| Method | PSNR | SSIM | Edge F1 |
|---|---:|---:|---:|
| Histogram Equalization | 8.06 | 0.2478 | 0.0517 |
| CLAHE | 8.43 | 0.2689 | 0.0788 |
| Gamma | 8.47 | 0.3194 | 0.0754 |
| White Balance | 7.79 | 0.2774 | 0.0737 |

Observations:

- Gamma achieved the highest PSNR among the initial classical methods.
- Gamma also achieved the highest SSIM.
- CLAHE achieved the highest initial Edge F1.
- No single method was best on all three metrics.

These methods provided the classical reference for the learning-based approach.


#  Edge Preservation Evaluation

Edge preservation was included because underwater enhancement should not only improve brightness and contrast but should also preserve important image structures.

Canny edge detection was used to extract edges.

The Edge F1 score was calculated using:

- Precision
- Recall
- F1 score

The initial Edge F1 results were:

- Histogram Equalization: 0.0517
- CLAHE: 0.0788
- Gamma: 0.0754
- White Balance: 0.0737

CLAHE produced the highest Edge F1 in the initial classical experiment.


#  Day 14: Visual Comparison

Visual comparisons were performed between:

- Input
- Target
- Histogram Equalization
- CLAHE
- Gamma
- White Balance

The purpose was to compare the visual appearance of the enhancement methods.

The comparison helped identify:

- Contrast improvement
- Brightness changes
- Color changes
- Detail preservation
- Difficult underwater scenes

Visual comparison was considered together with the numerical metrics.


#  Day 15: Baseline Review and Mentor Feedback

The classical baseline results were reviewed before moving to the learning-based stage.

The review focused on:

- PSNR
- SSIM
- Edge F1
- Visual quality
- Suitability of the classical methods
- Possible over-enhancement
- Preservation of image details

The classical methods were retained as reference methods for comparison with the CNN.


#  Day 16: Learning-Based Enhancement

A lightweight CNN was implemented for paired image-to-image enhancement.

The CNN learns the mapping:

Degraded Image → Enhanced Image

using the paired target images.


#  CNN Architecture

The CNN contains three convolution layers.

Architecture:

```text
Input RGB Image
↓
Conv2D: 3 → 32
↓
ReLU
↓
Conv2D: 32 → 32
↓
ReLU
↓
Conv2D: 32 → 3
↓
Sigmoid
↓
Enhanced RGB Image
```

The model was intentionally kept lightweight.

The final Sigmoid layer produces pixel values in the range:

0–1


#  CNN Training Configuration

Framework:

PyTorch

Optimizer:

Adam

Loss function:

Mean Squared Error (MSE)

Learning rate:

0.001

Batch size:

8

Hardware:

CUDA GPU / T4

The model was trained using the paired training dataset.

Training loss and validation loss were monitored during training.


#  Training and Validation

During training:

- The model receives an underwater input image.
- The CNN generates an enhanced output.
- The output is compared with the target.
- MSE loss is calculated.
- Backpropagation updates the model parameters.

During validation:

- The model is evaluated without updating its parameters.
- Validation loss is calculated.
- Training and validation behavior is monitored.

The purpose of validation is to help compare model configurations and monitor learning behavior.


#  Day 17: Method Refinement

After the initial experiments, parameter refinement was performed.

The main classical refinement focused on Gamma Correction.

Different Gamma configurations were tested.


#  Gamma Refinement Results

The tested Gamma configurations produced:

| Configuration | PSNR | SSIM | Edge F1 |
|---|---:|---:|---:|
| Gamma 1 | 6.9401 | 0.2126 | 0.0697 |
| Gamma 2 | 7.7274 | 0.2768 | 0.0718 |
| Gamma 3 | 8.4427 | 0.3189 | 0.0728 |
| Gamma 4 | 9.4528 | 0.3626 | 0.0719 |

Gamma 4 was selected as the Refined Gamma reference because it produced the highest PSNR and SSIM among the tested Gamma configurations.


#  CNN Refinement

CNN experiments were also performed using different learning-rate configurations.

Validation loss was used to compare the CNN configurations.

The selected CNN configuration was then used for final testing.

The refinement stage was used to make the final comparison more meaningful instead of comparing the CNN only with the initial Gamma configuration.


#  Day 18–20: Model Training and Evaluation

The selected CNN was trained using the paired training data.

Training and validation behavior were monitored.

The model was then evaluated on the held-out test set.

The test set was not used for model training.

This provided an independent evaluation of the final CNN.


#  Day 21–24: Final Evaluation and Refinement

During this stage, the final model outputs were evaluated using:

- PSNR
- SSIM
- Edge F1

The final CNN was also compared with the Refined Gamma reference.

The purpose was to determine whether the learning-based method provided improvement over the refined classical approach.


#  Final Results

Final comparison:

| Method | PSNR | SSIM | Edge F1 | Complexity |
|---|---:|---:|---:|---|
| Refined Gamma | 9.4528 | 0.3626 | 0.0719 | Low |
| Final CNN | 13.4626 | 0.4594 | 0.0147 | Higher |

Final CNN results:

PSNR:

13.4626 dB

SSIM:

0.4594

Edge F1:

0.0147


#  Final Result Interpretation

The final CNN achieved substantially higher PSNR and SSIM compared with the Refined Gamma reference.

Comparison:

PSNR improvement:

13.4626 vs 9.4528

SSIM improvement:

0.4594 vs 0.3626

However, the CNN achieved a lower Edge F1:

0.0147 vs 0.0719

This indicates that the metrics measure different aspects of image quality.

PSNR focuses on pixel-level reconstruction quality.

SSIM focuses on structural similarity.

Edge F1 focuses specifically on edge agreement.

Therefore, the final evaluation considers all three metrics instead of using only one metric.


#  Inference Time

CNN inference time was also measured.

Total test inference time:

0.2333 seconds

Number of test images:

23

Average inference time:

0.0101 seconds/image

This measurement provides information about the computational cost of applying the CNN to test images.


#  Day 25: Worst Prediction Analysis

Failure analysis was performed to understand where the CNN performed poorly.

For every test image:

1. CNN prediction was generated.
2. MSE between prediction and target was calculated.
3. Predictions were sorted according to MSE.
4. The worst five predictions were selected.
5. Input, CNN output and target images were visually compared.

The purpose was to identify difficult underwater scenes where the CNN did not produce a satisfactory enhancement.

This step is important because average metrics alone cannot show individual failure cases.


#  Day 26: Final Software Package and Report Preparation

The final stage focused on organizing the complete project.

The final project includes:

- Dataset preparation
- Preprocessing pipeline
- Classical enhancement methods
- Classical metrics
- Edge preservation evaluation
- CNN model
- CNN training
- Validation
- Parameter refinement
- Final evaluation
- Inference-time measurement
- Worst-case analysis
- Final result summary


# Final Project Pipeline

The complete pipeline is:

```text
Teacher-Provided Dataset
↓
Dataset Inspection
↓
Input-Target Pairing
↓
RGB Conversion
↓
256×256 Resize
↓
0–1 Normalization
↓
Train / Validation / Test Split
↓
Classical Enhancement
↓
PSNR / SSIM / Edge F1
↓
CNN Image-to-Image Model
↓
Training
↓
Validation
↓
Parameter Refinement
↓
Final Test Evaluation
↓
Inference Time Measurement
↓
Worst Prediction Analysis
↓
Final Report and Presentation
```


#  Technologies Used

Programming Language:

- Python

Libraries:

- PyTorch
- NumPy
- OpenCV
- Pillow
- Matplotlib
- scikit-image

Environment:

- Google Colab

Hardware:

- T4 GPU / CUDA


#  Evaluation Metrics

## PSNR

Peak Signal-to-Noise Ratio measures the pixel-level similarity between the enhanced image and target.

Higher PSNR generally indicates lower reconstruction error.

## SSIM

Structural Similarity Index measures structural similarity between the enhanced image and target.

Higher SSIM indicates greater structural similarity.

## Edge F1

Edge F1 evaluates how well the edges of the enhanced image match the edges of the target image.

It combines precision and recall into a single score.


