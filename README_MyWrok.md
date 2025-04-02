# EfficientNet Model for Deepfake Detection

## Project Overview

This project implements an EfficientNet-based deep learning model to classify images as either real or deepfake (AI-generated). The model was trained on a dataset containing 140,002 training images, 39,428 validation images, and 10,905 test images.

## Key Features

- Utilizes EfficientNetB0 as the base architecture with custom dense layers
- Implements data augmentation techniques to improve model generalization
- Achieves high validation accuracy (94.79% after 8 epochs)
- Includes comprehensive training visualization (accuracy/loss plots)
- Designed to run on GPU hardware for faster training

## Technical Implementation

### Model Architecture

The model uses transfer learning with:
- EfficientNetB0 base (pretrained on ImageNet)
- Global average pooling layer
- Three dense layers (512, 256, 128 units) with ReLU activation and batch normalization
- Final sigmoid output layer for binary classification

### Training Process

- Optimizer: Adam with learning rate 0.0001
- Loss function: Binary crossentropy
- Batch size: 32
- Epochs: 8
- Steps per epoch: 1024
- Data augmentation includes rotation, shifts, shear, zoom, and flipping

### Dataset Structure

The dataset is organized into three directories:
- Train: 140,002 images
- Validation: 39,428 images
- Test: 10,905 images

## Results

The model achieved:
- Training accuracy: 97.20%
- Validation accuracy: 94.79%
- Validation loss: 0.1571



This project demonstrates the effectiveness of EfficientNet for deepfake detection while maintaining computational efficiency through transfer learning and data augmentation techniques.
