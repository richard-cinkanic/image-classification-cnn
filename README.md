# Image Classification with CNNs

Image classification using a custom Convolutional Neural Network (CNN) implemented in PyTorch.

## Overview

This project was developed as part of the *Programming in Python II* course. The objective was to classify images into 20 object categories using a convolutional neural network trained entirely from scratch.

The challenge involved training a model on more than 12,000 labeled images and evaluating its performance on a hidden test set. Transfer learning was not allowed, requiring the architecture, training pipeline, and evaluation workflow to be implemented independently.

The workflow consists of:

1. Image preprocessing
2. Dataset preparation
3. CNN architecture design
4. Model training
5. Validation
6. Model checkpointing
7. Hidden test set evaluation

## Requirements

* Python 3.11
* PyTorch
* torchvision
* NumPy
* pandas
* Pillow
* scikit-learn

## Dataset

The dataset contains **12,483 labeled images** distributed across **20 object categories**.

Example classes:

* Book
* Bottle
* Car
* Cat
* Chair
* Cup
* Dog
* Flower
* Laptop
* Shoes
* Tree

Images were resized to a maximum dimension of 100 pixels while preserving aspect ratio.

The original dataset is not included in this repository.

## Data Processing

The preprocessing pipeline includes:

* Image loading
* Grayscale conversion
* Image resizing
* Padding to 100×100 pixels
* Tensor conversion
* Normalization

Pipeline:

```text
Input Images
      ↓
Grayscale Conversion
      ↓
Resize & Padding
      ↓
Tensor Conversion
      ↓
Normalization
      ↓
CNN Input
```

## Model Architecture

The model uses a custom convolutional neural network implemented in PyTorch.

```text
Input (100×100)
      ↓
Conv + BatchNorm + ReLU
      ↓
Max Pooling
      ↓
Conv + BatchNorm + ReLU
      ↓
Max Pooling
      ↓
Conv + BatchNorm + ReLU
      ↓
Max Pooling
      ↓
Conv + BatchNorm + ReLU
      ↓
Max Pooling
      ↓
Fully Connected Layers
      ↓
20 Output Classes
```

Key components:

* Convolutional layers
* Batch normalization
* ReLU activations
* Max pooling
* Dropout regularization
* Fully connected classifier

## Training Configuration

* Adam optimizer
* Cross-entropy loss
* Validation split
* Early stopping
* Model checkpoint saving

The model was trained using mini-batch gradient descent and evaluated on a validation set to monitor generalization performance.

## Technologies

* Python
* PyTorch
* NumPy
* pandas
* Pillow
* scikit-learn

## Skills Demonstrated

* Deep Learning
* Computer Vision
* Convolutional Neural Networks
* Image Processing
* Data Preprocessing
* Model Training
* Hyperparameter Tuning
* Model Evaluation

```text
image-classification-cnn/
├── src/
│   ├── architecture.py
│   ├── dataset.py
│   └── train.py
└── README.md
```

## How to Run

### Install Dependencies

```bash
pip install torch torchvision numpy pandas pillow scikit-learn
```

### Train the Model

```bash
python src/train.py
```

### Load Trained Weights

```python
model.load_state_dict(torch.load("model.pth"))
```

## Results

Challenge Leaderboard Accuracy: **52.9%**

The model was evaluated on a hidden test set provided through the course challenge server and achieved an accuracy of **52.9%** across 20 image classes.

The assignment awarded full points for models achieving at least 50% accuracy on the hidden test set.

## Notes

The original dataset is not included in this repository.

The trained model weights are not included due to file size limitations.

This repository focuses on the implementation of the CNN architecture, image preprocessing pipeline, training procedure, and evaluation workflow used for the image classification challenge.
