# FashionMNIST Image Classification using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the FashionMNIST dataset. The model learns to recognize 10 categories of clothing items such as shirts, trousers, sneakers, bags, and coats.

The project was built to strengthen understanding of deep learning concepts including convolutional layers, feature extraction, batch normalization, data augmentation, and mini-batch training.

---

## Dataset

FashionMNIST is a benchmark computer vision dataset consisting of:

* 60,000 training images
* 10,000 test images
* 10 clothing categories
* Grayscale images of size 28×28

Classes:

* T-shirt/Top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle Boot

---

## Model Architecture

CNN Architecture:

* Conv2D (1 → 32 channels)

* BatchNorm2D

* ReLU Activation

* Max Pooling

* Conv2D (32 → 64 channels)

* BatchNorm2D

* ReLU Activation

* Max Pooling

* Fully Connected Layer (3136 → 128)

* BatchNorm1D

* ReLU Activation

* Output Layer (128 → 10)

---

## Training Configuration

* Framework: PyTorch
* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Batch Size: 64
* Epochs: 7
* Device: GPU (CUDA) / CPU

### Data Augmentation

Training images were augmented using:

* Random Horizontal Flip
* Random Rotation (±10°)

---

## Results

| Metric            | Score      |
| ----------------- | ---------- |
| Test Accuracy     | **92.89%** |
| Training Accuracy | **91.72%** |

The model achieved strong performance while maintaining good generalization on unseen test data.

---

## Concepts Demonstrated

* Convolutional Neural Networks (CNNs)
* Feature Maps
* Max Pooling
* Batch Normalization
* Data Augmentation
* Mini-batch Gradient Descent
* Backpropagation
* GPU Training with PyTorch

---

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

---

## Future Improvements

* Learning Rate Scheduling
* Residual Networks (ResNet)
* Transfer Learning
* CIFAR-10 Classification
* Model Explainability and Visualization

