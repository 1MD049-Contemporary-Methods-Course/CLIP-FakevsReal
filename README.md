# CLIP Fine-Tuning for AI-Generated Content Detection

This repository contains Jupyter notebooks for fine-tuning OpenAI's CLIP model to detect AI-generated images, with implementations on two different datasets.

## Notebooks

### 1. CIFAKE Dataset (`CLIP_CIFAKE_FineTuning1.ipynb`)
Fine-tunes CLIP to distinguish between real and AI-generated images using the CIFAKE dataset (CIFAR-10 based).

**Key Features:**
- Zero-shot CLIP baseline evaluation
- Fine-tuned CLIP classifier achieving ~93.65% accuracy
- Comprehensive training pipeline with validation metrics
- Visualization of training progress and results
- Comparison between zero-shot and fine-tuned performance

### 2. Face Detection (`CLIP_Faces_FineTuning1.ipynb`)
Fine-tunes CLIP to detect real vs AI-generated human faces, with comparison to ResNet baseline.

**Key Features:**
- CLIP fine-tuning for face authenticity detection
- ResNet18 baseline for comparison
- Achieves ~79.46% validation accuracy with CLIP
- Detailed performance metrics (ROC-AUC, confusion matrix, classification reports)
- Training visualization and model comparison

## Datasets

### CIFAKE Dataset
A dataset containing 60,000 synthetic images (generated using Stable Diffusion 1.4) and 60,000 real images from CIFAR-10. The images are 32×32 pixels and span 10 different classes including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

- **Source:** [CIFAKE: Real and AI-Generated Synthetic Images on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Size:** 120,000 images total
- **Split:** 100,000 training + 20,000 test images
- **Classes:** Real vs Fake (AI-generated)

### Face Detection Dataset
A collection of real human faces and AI-generated synthetic faces for training deepfake detection models.

- **Source:** [140k Real and Fake Faces on Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Size:** 140,000 images total (70,000 real + 70,000 fake)
- **Real Images:** From Flickr-Faces-HQ (FFHQ) dataset
- **Fake Images:** Generated using StyleGAN
- **Resolution:** High-resolution face images

## Getting Started

### Running the Notebooks

**For CIFAKE Detection:**
1. Open `CLIP_CIFAKE_FineTuning1.ipynb`
2. Run all cells sequentially
3. The notebook will download the dataset automatically via kagglehub
4. View training metrics and compare zero-shot vs fine-tuned performance

**For Face Detection:**
1. Open `CLIP_Faces_FineTuning1.ipynb`
2. Run all cells to train both CLIP and ResNet models
3. The notebook will download the dataset automatically via kagglehub
4. Compare performance metrics between approaches

## Results Summary

| Model | Dataset | Validation Accuracy |
|-------|---------|-------------------|
| Fine-tuned CLIP | CIFAKE | ~93.65% |
| Fine-tuned CLIP | Faces | ~79.46% |
| ResNet18 | Faces | Variable (trained for comparison) |

## Technical Details

Both notebooks implement:
- Custom PyTorch datasets for loading images
- Fine-tuning of CLIP's visual encoder with frozen text encoder
- Linear classification head training
- Comprehensive evaluation metrics
- Training/validation split with proper data loading
