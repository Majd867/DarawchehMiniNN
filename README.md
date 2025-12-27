# DarawchehMiniNN

**DarawchehMiniNN** is a lightweight educational neural network library implemented from scratch using **NumPy**. 

**Goal** is to make neural networks applying much easier to save you time and effort.

This library is inspired by frameworks such as **PyTorch** and **TensorFlow**, but designed to be **minimal, transparent, and easy to extend**.


## Features

- Forward and backward propagation (manual backpropagation)
- Fully connected (Dense / Affine) layer
- Activation functions:
  - ReLU
  - Sigmoid
  - Tanh
  - Linear
- Loss functions:
  - Mean Squared Error (MSE)
  - Softmax Cross-Entropy
- Optimizers:
  - SGD
  - Momentum
  - AdaGrad
  - Adam
- Regularization and training helpers:
  - Dropout
  - Batch Normalization (BatchNorm1D)
- Trainer class for mini-batch training
- Hyperparameter tuning support (grid search)

## Installation (Local)

This library does not require installation as a package.  
Just make sure you have Python **3.10+** installed.

Install dependencies:

```bash
pip install -r requirements.txt
## How to run the example

From project root:

py -3.13 -m Darawcheh_mini_nn.examples.train_example_full
