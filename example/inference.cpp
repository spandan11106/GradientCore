#include "../gradientcore/include/gradientcore/base/arena.hpp"
#include "../gradientcore/include/gradientcore/graph.hpp"
#include "../gradientcore/include/gradientcore/matrix.hpp"
#include "../gradientcore/include/gradientcore/model.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace gradientcore;

namespace gradientcore {
void draw_mnist_digit(float *data);
void create_mnist_model(Arena *arena, model_context *model);
} // namespace gradientcore

int main() {
  Arena *perm_arena = Arena::create(MiB(128), MiB(1), false);

  matrix *test_images =
      mat_load(perm_arena, 10000, 784, "../data/test_images.mat");
  matrix *test_labels =
      mat_load(perm_arena, 10000, 1, "../data/test_labels.mat");

  model_context *model = model_create(perm_arena);
  create_mnist_model(perm_arena, model);
  model_compile(perm_arena, model);

  if (!model_load_weights(model, "model.bin")) {
    printf("Failed to load model.bin");
    return 1;
  }

  int num = 0;
  printf("Enter a number ( < 10000): ");
  std::cin >> num;

  uint32_t images_index = num % 10000;
  float *selected_image = test_images->data + (images_index * 784);

  memcpy(model->input->val->data, selected_image, sizeof(float) * 784);

  model_feedforward(model);
  draw_mnist_digit(selected_image);

  uint32_t prediction = mat_argmax(model->output->val);
  uint32_t expected_output = (uint32_t)test_labels->data[images_index];
  float confidence = model->output->val->data[prediction] * 100.0f;

  printf("Model prediction: %d, Expected output: %d, Confidence: %.2f%% \n",
         prediction, expected_output, confidence);
}
