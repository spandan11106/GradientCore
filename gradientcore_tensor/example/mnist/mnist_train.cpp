// MNIST Training Example using GradientCore Tensor API
//
// Network:  784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
// Loss:     Cross Entropy (mean)
// Optim:    Adam (lr=0.001)
// Data:     Raw float32 .mat files from data/

#include "../../../gradientcore/include/gradientcore/base/prng.hpp"
#include "../../include/gradientcore/autograd/autograd.hpp"
#include "../../include/gradientcore/autograd/engine.hpp"
#include "../../include/gradientcore/core/arena.hpp"
#include "../../include/gradientcore/nn/nn.hpp"
#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/optim/optim.hpp"
#include "../../include/gradientcore/serialize/serialize.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace gradientcore;

// Data Loading
struct MNISTData {
  float *images; // [num_samples, 784]
  float *labels; // [num_samples] (digit 0-9)
  uint32_t num_samples;
};

static MNISTData load_data(const char *images_path, const char *labels_path,
                           uint32_t expected_count) {
  MNISTData data = {};
  data.num_samples = expected_count;

  FILE *f_img = std::fopen(images_path, "rb");
  if (!f_img) {
    std::fprintf(stderr, "Error: cannot open %s\n", images_path);
    std::exit(1);
  }
  data.images = (float *)std::malloc(expected_count * 784 * sizeof(float));
  std::fread(data.images, sizeof(float), expected_count * 784, f_img);
  std::fclose(f_img);

  FILE *f_lbl = std::fopen(labels_path, "rb");
  if (!f_lbl) {
    std::fprintf(stderr, "Error: cannot open %s\n", labels_path);
    std::exit(1);
  }
  data.labels = (float *)std::malloc(expected_count * sizeof(float));
  std::fread(data.labels, sizeof(float), expected_count, f_lbl);
  std::fclose(f_lbl);

  return data;
}

static void free_data(MNISTData *data) {
  std::free(data->images);
  std::free(data->labels);
}

// Visualization
static void draw_digit(float *pixel_data) {
  for (uint32_t y = 0; y < 28; y++) {
    for (uint32_t x = 0; x < 28; x++) {
      float num = pixel_data[x + y * 28];
      uint32_t col = 232 + (uint32_t)(num * 24);
      std::printf("\x1b[48;5;%dm  ", col);
    }
    std::printf("\x1b[0m\n");
  }
}

int main() {
  prng::seed(42, 1);

  std::printf("=== MNIST Training (GradientCore Tensor) ===\n\n");

  // Load data
  std::printf("Loading data...\n");
  MNISTData train = load_data("../../../data/train_images.mat",
                              "../../../data/train_labels.mat", 60000);
  MNISTData test = load_data("../../../data/test_images.mat",
                             "../../../data/test_labels.mat", 10000);
  std::printf("  Train: %u samples, Test: %u samples\n\n", train.num_samples,
              test.num_samples);

  // Show a sample digit
  std::printf("Sample digit (label = %.0f):\n", train.labels[0]);
  draw_digit(train.images);
  std::printf("\n");

  // Hyperparameters
  const uint32_t BATCH_SIZE = 32;
  const uint32_t EPOCHS = 3;
  const float LR = 0.001f;
  const uint32_t NUM_BATCHES = train.num_samples / BATCH_SIZE;

  // Create arena (256 MiB for MNIST)
  Arena *arena = Arena::create(MiB(256), MiB(4), true);
  GraphContext *ctx = graph_create(arena);

  // Define Network: 784 → 128 → 64 → 10
  nn::Linear layer1 = nn::linear_create(arena, ctx, 784, 128, true);
  nn::Linear layer2 = nn::linear_create(arena, ctx, 128, 64, true);
  nn::Linear layer3 = nn::linear_create(arena, ctx, 64, 10, true);

  Node *params[] = {layer1.weight, layer1.bias,   layer2.weight,
                    layer2.bias,   layer3.weight, layer3.bias};
  const uint32_t NUM_PARAMS = 6;

  // Create optimizer
  optim::Adam opt = optim::adam_create(arena, params, NUM_PARAMS, LR);

  // Pre-allocate input and target nodes (reused each batch)
  uint32_t input_shape[2] = {BATCH_SIZE, 784};
  Node *input = node_create(arena, ctx, 2, input_shape, NODE_FLAG_NONE);

  uint32_t target_shape[2] = {BATCH_SIZE, 10};
  Node *target_onehot =
      node_create(arena, ctx, 2, target_shape, NODE_FLAG_NONE);

  uint32_t base_num_nodes = ctx->num_nodes;
  uint64_t graph_start = arena->get_pos();

  // Training Loop
  for (uint32_t epoch = 0; epoch < EPOCHS; epoch++) {
    float epoch_loss = 0.0f;
    uint32_t epoch_correct = 0;

    for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
      uint32_t offset = batch * BATCH_SIZE;

      std::memcpy(input->val->storage->data, train.images + offset * 784,
                  BATCH_SIZE * 784 * sizeof(float));

      std::memset(target_onehot->val->storage->data, 0,
                  BATCH_SIZE * 10 * sizeof(float));
      for (uint32_t i = 0; i < BATCH_SIZE; i++) {
        int label = (int)train.labels[offset + i];
        target_onehot->val->storage->data[i * 10 + label] = 1.0f;
      }

      if (epoch == 0 && batch == 0) {
        arena->pop_to(graph_start);
        ctx->num_nodes = base_num_nodes;
        Node *h1 = nn::linear_forward(arena, ctx, layer1, input);
        h1 = node_relu(arena, ctx, h1);
        Node *h2 = nn::linear_forward(arena, ctx, layer2, h1);
        h2 = node_relu(arena, ctx, h2);
        Node *logits = nn::linear_forward(arena, ctx, layer3, h2);
        Node *probs = node_softmax(arena, ctx, logits);

        std::printf("pre-training output: \n");
        for (uint32_t i = 0; i < 10; i++) {
          std::printf("%.2f ", probs->val->storage->data[i]);
        }
        std::printf("\n\n");
      }

      // Reset computation graph
      arena->pop_to(graph_start);
      ctx->num_nodes = base_num_nodes;

      // Zero gradients
      optim::zero_grad(params, NUM_PARAMS);

      // Forward pass
      Node *h1 = nn::linear_forward(arena, ctx, layer1, input);
      h1 = node_relu(arena, ctx, h1);

      Node *h2 = nn::linear_forward(arena, ctx, layer2, h1);
      h2 = node_relu(arena, ctx, h2);

      Node *logits = nn::linear_forward(arena, ctx, layer3, h2);
      Node *probs = node_softmax(arena, ctx, logits);

      // Cross entropy loss
      Node *ce = node_cross_entropy(arena, ctx, target_onehot, probs);
      Node *loss = node_mean(arena, ctx, ce);

      // Backward
      GraphProgram prog = graph_compile(arena, ctx, loss);
      graph_backward(&prog);

      // Update weights
      optim::adam_step(&opt);

      epoch_loss += loss->val->storage->data[0];

      // Count correct predictions in this batch
      for (uint32_t i = 0; i < BATCH_SIZE; i++) {
        int pred = 0;
        float max_prob = probs->val->storage->data[i * 10];
        for (int c = 1; c < 10; c++) {
          float p = probs->val->storage->data[i * 10 + c];
          if (p > max_prob) {
            max_prob = p;
            pred = c;
          }
        }
        if (pred == (int)train.labels[offset + i])
          epoch_correct++;
      }

      if ((batch + 1) % 200 == 0) {
        float avg_loss = epoch_loss / (batch + 1);
        float acc = 100.0f * (float)epoch_correct / ((batch + 1) * BATCH_SIZE);
        std::printf("  Epoch %u | Batch %4u/%u | Loss: %.4f | Acc: %.1f%%\n",
                    epoch + 1, batch + 1, NUM_BATCHES, avg_loss, acc);
      }
    }

    float avg_loss = epoch_loss / NUM_BATCHES;
    float train_acc =
        100.0f * (float)epoch_correct / (NUM_BATCHES * BATCH_SIZE);
    std::printf("Epoch %u Complete | Avg Loss: %.4f | Train Acc: %.1f%%\n\n",
                epoch + 1, avg_loss, train_acc);
  }

  std::memcpy(input->val->storage->data, train.images,
              BATCH_SIZE * 784 * sizeof(float));
  arena->pop_to(graph_start);
  ctx->num_nodes = base_num_nodes;
  {
    Node *h1 = nn::linear_forward(arena, ctx, layer1, input);
    h1 = node_relu(arena, ctx, h1);
    Node *h2 = nn::linear_forward(arena, ctx, layer2, h1);
    h2 = node_relu(arena, ctx, h2);
    Node *logits = nn::linear_forward(arena, ctx, layer3, h2);
    Node *probs = node_softmax(arena, ctx, logits);

    std::printf("post-training output: \n");
    for (uint32_t i = 0; i < 10; i++) {
      std::printf("%.2f ", probs->val->storage->data[i]);
    }
    std::printf("\n\n");
  }

  std::printf("Saving trained weights to model.bin...\n");
  if (serialize::save_weights("model.bin", params, NUM_PARAMS)) {
    std::printf("Successfully saved !!\n");
  } else {
    std::printf("Failed to save the model.\n");
  }

  // Cleanup
  free_data(&train);
  free_data(&test);
  arena->destroy();

  return 0;
}
