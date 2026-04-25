# Colorization of CIFAR-10 Horse Images using CNNs

This repository contains the implementation of **three different convolutional neural network (CNN) architectures** for grayscale-to-color image colorization using only the **horse class** from the **CIFAR-10** dataset.
---

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Required Libraries](#required-libraries)
- [Techniques Used](#techniques-used)
  - [Data Preparation](#data-preparation)
  - [Model Architectures](#model-architectures)
- [Training and Evaluation](#training-and-evaluation)
- [Visualization](#visualization)
- [Results and Outputs](#results-and-outputs)
  - [Regression Model Performance](#regression-model-performance)
  - [Classification Models Performance](#classification-models-performance)
- [Summary](#summary)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## Overview

Image colorization is the task of predicting the missing color channels of a grayscale image. In this project, the focus is on **horse images from CIFAR-10**.

The pipeline includes:

- **Data loading & preprocessing**: extracting only the horse class, converting RGB images to grayscale, and creating color labels via **K-Means clustering**.
- **Model building**: implementing three CNN-based architectures:
  - **RegressionCNN** – directly predicts RGB pixel values.
  - **PixelClassificationCNN** – predicts one of 24 color bins per pixel using a custom `MyConv2d` layer.
  - **UNetPixelClassifier** – a U-Net with skip connections for improved spatial consistency.
- **Training & evaluation**: training each model, tracking validation performance, and comparing final test results.
- **Visualization**: displaying grayscale inputs, ground-truth images, predicted outputs, and training loss curves.

The best models are selected based on **validation loss**, and their final performance is reported on a **held-out test set**.

---

## Installation & Setup

1. **Clone the repository** (or download the notebook).
2. Create and activate a **Python virtual environment** (recommended).
3. Install the required dependencies:

```bash
pip install numpy matplotlib tqdm torch torchvision scikit-learn
```

4. Run the notebook:

```bash
M_Talha_K257605_Assignment_03.ipynb
```

You can open it using:
- **Jupyter Notebook**
- **Jupyter Lab**
- **Google Colab**

> **Note:** If using Google Colab, make sure to enable **GPU runtime**:
> `Runtime -> Change runtime type -> GPU`

---

## Required Libraries

The notebook uses standard Python libraries for deep learning, visualization, and image processing.

### Main Dependencies

- `numpy`
- `matplotlib`
- `tqdm`
- `torch`
- `torchvision`
- `scikit-learn`

### Explicit Imports Used

- `os`, `sys`, `math`, `time`, `json`, `random`
- `pathlib`, `typing`, `dataclasses`
- `numpy`, `matplotlib.pyplot`, `tqdm`
- `torch`, `torch.nn`, `torch.nn.functional`
- `torch.utils.data` (`Dataset`, `DataLoader`, `Subset`)
- `torchvision.datasets.CIFAR10`
- `torchvision.transforms.functional`
- `sklearn.cluster.KMeans`

---

## Techniques Used

### Data Preparation

The dataset pipeline includes the following steps:

- **Filtering**: only the **horse class** (CIFAR-10 label `7`) is extracted from the full training and test sets.
- **Grayscale conversion**: RGB images are converted to single-channel grayscale using the luminosity formula:

```text
0.2989 R + 0.5870 G + 0.1140 B
```

- **Color quantization**: **K-Means clustering** with **24 bins** is applied to all RGB pixels from the horse training images.
- Each pixel is assigned to its nearest cluster center, producing a pixel-wise **class map** of shape `[H, W]` with **24 possible classes**.
- These class maps are used as targets for the classification-based models.
- **Train/validation split**: the horse training subset is divided into:
  - **4,000 images for training**
  - **1,000 images for validation**

---

### Model Architectures

#### 1. RegressionCNN

A simple encoder-decoder CNN that directly predicts RGB values.

**Characteristics:**
- Uses **DoubleConv blocks**:
  - Convolution
  - Batch Normalization
  - ReLU
  - repeated twice
- No skip connections
- Outputs **3 channels** corresponding to RGB
- Uses **sigmoid activation** in the final layer
- Optimized using **Mean Squared Error (MSE)**

---

#### 2. PixelClassificationCNN

A CNN that treats colorization as a **pixel-wise classification problem**.

**Characteristics:**
- Shares the same encoder-decoder backbone as `RegressionCNN`
- Replaces standard convolution with a custom layer called **`MyConv2d`**
- `MyConv2d` is implemented manually using:
  - `torch.nn.functional.unfold`
  - `torch.einsum`
- Supports `groups=1` only
- Outputs **24 logits per pixel**
- Uses **Cross-Entropy Loss**

**Inference:**
- Predicted logits are transformed into RGB values using a **probability-weighted sum** of the 24 learned color centers.

---

#### 3. UNetPixelClassifier

A U-Net based pixel classifier that improves spatial consistency through skip connections.

**Characteristics:**
- U-Net style encoder-decoder architecture
- **Skip connections** concatenate encoder feature maps with decoder activations
- The **input grayscale image** is also concatenated at the final stage to preserve fine details
- Uses the same custom **`MyConv2d`** layer
- Outputs **24 classes per pixel**
- Optimized using **Cross-Entropy Loss**

---

## Training and Evaluation

All models are trained using the same optimizer configuration:

- **Optimizer**: Adam
- **Learning rate**: `1e-3`
- **Weight decay**: `1e-5`

### Training Schedule

- **RegressionCNN** was trained for:
  - **5 epochs**
  - **10 epochs**
  - **20 epochs**

- **PixelClassificationCNN** and **UNetPixelClassifier** were trained for:
  - **20 epochs** each

### Evaluation

A separate evaluation function is used to compute:
- **Validation loss**
- **Test loss**

### Monitored Metrics

- **Training loss per epoch**
- **Validation loss per epoch**

---

## Visualization

The notebook includes several visual outputs:

- **Training curves** (loss vs. epoch) for each model
- **Colorization triplets** showing:
  1. Input grayscale image
  2. Ground truth RGB image
  3. Predicted colorized output

These visualizations allow qualitative comparison between the regression and classification approaches.

---

## Results and Outputs

### Regression Model Performance

| Epochs | Validation MSE | Test MSE |
|--------|----------------|----------|
| 5      | 0.00755        | 0.00774  |
| 10     | 0.00622        | 0.00635  |
| 20     | 0.00653        | 0.00670  |

**Observations:**
- The model trained for **10 epochs** achieved the **lowest validation MSE**.
- Training for **20 epochs** showed signs of **overfitting**, as validation loss stopped improving and fluctuated.
- Therefore, the **10-epoch model** was selected as the best regression model.

---

### Classification Models Performance

| Model | Validation Cross-Entropy | Test Cross-Entropy |
|------|---------------------------|--------------------|
| PixelClassificationCNN | 1.8080 | 1.8432 |
| UNetPixelClassifier | 1.1403 | 1.1445 |

**Observations:**
- The **UNetPixelClassifier** significantly outperformed the plain `PixelClassificationCNN`.
- Skip connections and final grayscale concatenation helped preserve structure and fine image details.
- Both classification models produced **sharper and more visually pleasing colorizations** than the regression approach.

---

## Summary

A final summary dictionary containing the best-performing models and evaluation metrics is saved as:

```json
{
  "best_regression_epochs": 10,
  "best_regression_val": {"loss": 0.0062152879312634465},
  "best_regression_test": {"loss": 0.006345843197777867},
  "classification_val": {"loss": 1.8080033302307128},
  "classification_test": {"loss": 1.8431748509407044},
  "unet_val": {"loss": 1.140299654006958},
  "unet_test": {"loss": 1.1445401310920715}
}
```

### Key Takeaways

- The **UNet pixel classifier** is the overall **best-performing model**.
- It achieved the **lowest test loss** among all approaches.
- It also produced the **most coherent and realistic colorizations** qualitatively.
- The regression model performed reasonably well but produced smoother and less vivid outputs.

---

## Repository Structure

```text
.
├── M_Talha_K257605_Assignment_03.ipynb   # Main assignment notebook
├── README.md                             # Project documentation
└── figures/                              # Output directory (checkpoints, figures)
```

---

## License

This project is intended for **educational purposes only** as part of an academic assignment. All external libraries and tools are used under their respective licenses.
