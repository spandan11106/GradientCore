# MNIST Example (GradientCore Tensor)

This example demonstrates how to build and train a Multi-Layer Perceptron (MLP) for digit classification using the improved `gradientcore_tensor` API. 

The network architecture is `784 → 128 → 64 → 10`, using ReLU activations, Cross-Entropy loss, and the Adam Optimizer.

### Performance
Even on a purely unoptimized C++ CPU implementation, the model trains cleanly and converges rapidly. It achieves **~98% accuracy** on the test set in just **3 epochs**.

## How to Run

1. **Train the Model:**
   This command will run the training loop for 3 epochs and save the resulting weights to `model.bin`.
   ```bash
   make run
   ```

2. **Run Inference:**
   Once `model.bin` is generated, you can run the interactive inference script. This will prompt you to enter a number (0-9999). It will then display the corresponding MNIST test digit visually in your terminal, and output the model's prediction and confidence.
   ```bash
   make run-inference
   ```

3. **Clean Up:**
   When you're done, or if you want to rebuild the project artifacts:
   ```bash
   make clean
   ```

Enjoy testing out the tensor auto-differentiation engine!
