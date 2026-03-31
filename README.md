# GradientCore

A high-performance, modular C++ neural network library with automatic differentiation and computational graph support.

## Overview

GradientCore is a lightweight automatic differentiation framework built from scratch in C++11. It provides a flexible foundation for building and training neural networks with support for forward propagation, backward propagation (gradient computation), and optimized training loops.

## Inspiration

This project is inspired by [Magicalbat](https://github.com/Magicalbat)'s original implementation of an automatic differentiation engine in C. GradientCore reimplements the same core concepts and architecture in modern C++, leveraging the language's features for cleaner abstractions and better maintainability while preserving the efficiency and minimalist philosophy of the original.

## What You Get

- **Automatic Differentiation**: Compute gradients automatically using a computational graph approach
- **Modular Architecture**: Clean separation of concerns across matrix operations, operators, graph computation, and training
- **Memory Efficient**: Custom arena allocator for fast memory management
- **Cross-Platform**: Platform abstraction layer supporting Linux and Windows
- **Lightweight**: No external dependencies for the core library (example uses TensorFlow for data loading)

## Example Application: MNIST Digit Classification

The `example/` folder contains a complete MNIST digit classification application demonstrating:
- Loading training and test datasets (60,000 training images, 10,000 test images)
- Creating a neural network with hidden layers and ReLU activations
- Training with cross-entropy loss and SGD optimizer
- Evaluating accuracy on test data

### Getting Started

#### 1. Prepare the Dataset

First, download and prepare the MNIST dataset:

```bash
cd example/
python enlist.py
```

This script uses TensorFlow datasets to download MNIST and saves it in binary format to the `data/` folder. The dataset includes:
- `train_images.mat` and `train_labels.mat` (60,000 samples)
- `test_images.mat` and `test_labels.mat` (10,000 samples)

#### 2. Build the Project

The project uses a unified Makefile that compiles all library sources modularly and links them into a single executable:

```bash
cd example/
make clean    # Clean previous build artifacts
make          # Compile and link (creates mnist_example)
```

The Makefile produces the final `mnist_example` executable

#### 3. Train the Model

Run the training application:

```bash
cd example/
make run
```

This trains a neural network on MNIST digits for 3 epochs. The model achieves **~94% accuracy** on the test set.


## Project Structure

```
GradientCore/
├── gradientcore/              # Core library
│   ├── include/               # Public headers
│   │   └── gradientcore/
│   │       ├── matrix.hpp
│   │       ├── model.hpp
│   │       ├── operators.hpp
│   │       ├── graph.hpp
│   │       ├── training.hpp
│   │       └── base/          # Utility headers
│   └── src/                   # Implementation (modular)
│       ├── matrix/            # 4 files: ops, arithmetic, multiply, activations
│       ├── operators/         # 2 files: unary, binary
│       ├── graph/             # 3 files: topological, compute, compile
│       ├── training/          # 1 file: SGD training loop
│       ├── base/              # Arena allocator and PRNG
│       └── platform/          # Platform-specific code
├── example/                   # MNIST example application
│   ├── main.cpp               # Application entry point
│   ├── mnist.cpp              # MNIST-specific helpers
│   ├── enlist.py              # Dataset download script
│   ├── Makefile               # Build configuration
│   └── README.md              # Example usage guide
└── data/                      # Training and test data
```

## Future Enhancements

- Additional activation functions (Tanh, Sigmoid, ELU)
- More optimization algorithms (Adam, RMSprop)
- Batch normalization support
- Convolution operations

## License

See [LICENSE](LICENSE) file for details.
