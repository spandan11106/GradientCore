#include "../gradientcore/include/gradientcore/base/arena.hpp"
#include "../gradientcore/include/gradientcore/graph.hpp"
#include "../gradientcore/include/gradientcore/matrix.hpp"
#include "../gradientcore/include/gradientcore/model.hpp"
#include "../gradientcore/include/gradientcore/training.hpp"

#include <cstdio>
#include <cstring>

using namespace gradientcore;

// Forward declarations for MNIST utilities
namespace gradientcore {
void draw_mnist_digit(float *data);
void create_mnist_model(Arena *arena, model_context *model);
} // namespace gradientcore

int main() {
  Arena *perm_arena = Arena::create(MiB(1024), MiB(1), false);

  matrix *train_images =
      mat_load(perm_arena, 60000, 784, "../data/train_images.mat");
  matrix *test_images =
      mat_load(perm_arena, 10000, 784, "../data/test_images.mat");
  matrix *train_labels = mat_create(perm_arena, 60000, 10);
  matrix *test_labels = mat_create(perm_arena, 10000, 10);

  {
    matrix *train_labels_file =
        mat_load(perm_arena, 60000, 1, "../data/train_labels.mat");
    matrix *test_labels_file =
        mat_load(perm_arena, 10000, 1, "../data/test_labels.mat");

    for (uint32_t i = 0; i < 60000; i++) {
      uint32_t num = train_labels_file->data[i];
      train_labels->data[i * 10 + num] = 1.0f;
    }

    for (uint32_t i = 0; i < 10000; i++) {
      uint32_t num = test_labels_file->data[i];
      test_labels->data[i * 10 + num] = 1.0f;
    }
  }

  draw_mnist_digit(train_images->data);
  for (uint32_t i = 0; i < 10; i++) {
    printf("%.0f ", train_labels->data[i]);
  }
  printf("\n\n");

  model_context *model = model_create(perm_arena);
  create_mnist_model(perm_arena, model);
  model_compile(perm_arena, model);

  memcpy(model->input->val->data, train_images->data, sizeof(float) * 784);

  model_feedforward(model);

  printf("pre-training output: \n");
  for (uint32_t i = 0; i < 10; i++) {
    printf("%.2f ", model->output->val->data[i]);
  }

  printf("\n\n");

  model_training_desc training_desc = {
      .train_images = train_images,
      .train_labels = train_labels,
      .test_images = test_images,
      .test_labels = test_labels,

      .epochs = 100,
      .batch_size = 50,
      .learning_rate = 0.01f,
  };

  model_train(model, &training_desc);

  memcpy(model->input->val->data, train_images->data, sizeof(float) * 784);

  model_feedforward(model);

  printf("post-training output: \n ");
  for (uint32_t i = 0; i < 10; i++) {
    printf("%.2f ", model->output->val->data[i]);
  }

  printf("\n");

  printf("Saving trained weights to model.bin...\n");
  if (model_save_weights(model, "model.bin")) {
    printf("Successfully saved !!\n");
  } else {
    printf("Failed to save the model.\n");
  }

  perm_arena->destroy();
  return 0;
}
