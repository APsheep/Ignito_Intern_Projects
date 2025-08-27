# Handwritten Digit Recognition (MNIST)
# Overview

This project implements a neural network in TensorFlow/Keras to classify handwritten digits from the MNIST dataset. Despite its simple architecture, the model achieves ~97% accuracy on unseen test data, showing the power of even small deep learning models.

# Dataset

Source: MNIST dataset (available directly in Keras)

Images: 70,000 grayscale handwritten digits (28×28 pixels)

60,000 training images

10,000 test images

Classes: Digits 0–9 (10 classes)

# Methodology

# Preprocessing

Normalize pixel values (0–255 → 0–1 range)

One-hot encode class labels (e.g., 3 → [0,0,0,1,0,0,0,0,0,0])

Model Architecture (Sequential)

Input: 28×28 pixels

Flatten: 784-element vector

Dense(64, activation='relu'): hidden layer

Dense(10, activation='softmax'): output layer for class probabilities

# Compilation

Optimizer: Adam (Adaptive Moment Estimation)

Loss: Categorical Crossentropy (multi-class classification)

Metric: Accuracy

# Training

5 epochs

20% of training data reserved for validation

Backpropagation + gradient updates with Adam optimizer

# Results

Test Accuracy: ~97%

Classification Report: Precision, recall, F1-scores all >96% for most classes

Confusion Matrix: Minor misclassifications (e.g., 5 vs. 8 due to shape similarity)

Even with a simple 2-layer network, performance is excellent.

[Demo Video](https://drive.google.com/file/d/18gS5aBF7bVFF0cLVuUrIZGf-W9VaVBSa/view?usp=drive_link)

This project demonstrates the fundamentals of deep learning for image classification and provides a solid foundation for scaling into more complex architectures like CNNs.
