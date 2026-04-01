// MNIST Inference Example using GradientCore Tensor API
//
// Loads a saved model and runs inference on user-selected test images.

#include "../../include/gradientcore/autograd/autograd.hpp"
#include "../../include/gradientcore/autograd/engine.hpp"
#include "../../include/gradientcore/core/arena.hpp"
#include "../../include/gradientcore/nn/nn.hpp"
#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/serialize/serialize.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

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
  std::printf("=== MNIST Inference (GradientCore Tensor) ===\n\n");

  // Create arena for inference (smaller, no need for large backward pass)
  Arena *arena = Arena::create(MiB(64), MiB(1), true);
  GraphContext *ctx = graph_create(arena);

  // Define Network: 784 → 128 → 64 → 10
  nn::Linear layer1 = nn::linear_create(arena, ctx, 784, 128, true);
  nn::Linear layer2 = nn::linear_create(arena, ctx, 128, 64, true);
  nn::Linear layer3 = nn::linear_create(arena, ctx, 64, 10, true);

  Node *params[] = {layer1.weight, layer1.bias,   layer2.weight,
                    layer2.bias,   layer3.weight, layer3.bias};
  const uint32_t NUM_PARAMS = 6;

  // Load weights
  std::printf("Loading weights model.bin...\n");
  if (!serialize::load_weights("model.bin", params, NUM_PARAMS)) {
    std::fprintf(stderr, "Failed to load mnist_model.gcw! Please run 'make "
                         "run_train' to generate it.\n");
    arena->destroy();
    return 1;
  }
  std::printf("Weights loaded successfully.\n\n");

  // Load test data
  std::printf("Loading test data...\n");
  MNISTData test = load_data("../../../data/test_images.mat",
                             "../../../data/test_labels.mat", 10000);
  std::printf("Loaded %u test samples.\n\n", test.num_samples);

  // Pre-allocate input node (batch size 1)
  uint32_t input_shape[2] = {1, 784};
  Node *input = node_create(arena, ctx, 2, input_shape, NODE_FLAG_NONE);

  uint32_t base_num_nodes = ctx->num_nodes;
  uint64_t graph_start = arena->get_pos();

  while (true) {
    int num = 0;
    std::printf("\nEnter a number (0-9999) or -1 to quit: ");
    if (!(std::cin >> num) || num == -1) {
      break;
    }

    if (num < 0 || num >= 10000) {
      std::printf("Invalid input. Please enter a number between 0 and 9999.\n");
      continue;
    }

    uint32_t images_index = num;
    float *selected_image = test.images + (images_index * 784);

    // Copy selected image to inference input
    std::memcpy(input->val->storage->data, selected_image, sizeof(float) * 784);

    // Reset computation graph to clear previous inference nodes
    arena->pop_to(graph_start);
    ctx->num_nodes = base_num_nodes;

    // Build inference graph
    Node *h1 = nn::linear_forward(arena, ctx, layer1, input);
    h1 = node_relu(arena, ctx, h1);

    Node *h2 = nn::linear_forward(arena, ctx, layer2, h1);
    h2 = node_relu(arena, ctx, h2);

    Node *logits = nn::linear_forward(arena, ctx, layer3, h2);
    Node *probs = node_softmax(arena, ctx, logits);

    // Display the image
    std::printf("\nImage [%u]:\n", images_index);
    draw_digit(selected_image);

    // Find the prediction
    int pred = 0;
    float max_prob = probs->val->storage->data[0];
    for (int c = 1; c < 10; c++) {
      float p = probs->val->storage->data[c];
      if (p > max_prob) {
        max_prob = p;
        pred = c;
      }
    }

    uint32_t expected_output = (uint32_t)test.labels[images_index];
    float confidence = max_prob * 100.0f;

    std::printf("\nModel prediction: %d | Expected output: %u | Confidence: "
                "%.2f%% %s\n",
                pred, expected_output, confidence,
                pred == (int)expected_output ? "✓" : "✗");
  }

  // Cleanup
  free_data(&test);
  arena->destroy();

  return 0;
}
