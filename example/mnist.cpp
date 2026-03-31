#include "../gradientcore/include/gradientcore/model.hpp"
#include "../gradientcore/include/gradientcore/matrix.hpp"
#include "../gradientcore/include/gradientcore/operators.hpp"
#include "../gradientcore/include/gradientcore/base/arena.hpp"

#include <cstdio>
#include <cmath>

namespace gradientcore {

void draw_mnist_digit(float *data) {
  for (uint32_t y = 0; y < 28; y++) {
    for (uint32_t x = 0; x < 28; x++) {
      float num = data[x + y * 28];
      uint32_t col = 232 + (uint32_t)(num * 24);
      printf("\x1b[48;5;%dm  ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m");
}

void create_mnist_model(Arena *arena, model_context *model) {
  model_var *input = mv_create(arena, model, 784, 1, MY_FLAG_INPUT);

  model_var *W0 = mv_create(arena, model, 128, 784,
                            MY_FLAG_REQUIRES_GRAD | MY_FLAG_PARAMETER);
  model_var *W1 = mv_create(arena, model, 128, 128,
                            MY_FLAG_REQUIRES_GRAD | MY_FLAG_PARAMETER);
  model_var *W2 = mv_create(arena, model, 10, 128,
                            MY_FLAG_REQUIRES_GRAD | MY_FLAG_PARAMETER);

  float bound0 = std::sqrt(6.0f / (784 + 128));
  float bound1 = std::sqrt(6.0f / (128 + 128));
  float bound2 = std::sqrt(6.0f / (128 + 10));
  mat_fill_rand(W0->val, -bound0, bound0);
  mat_fill_rand(W1->val, -bound1, bound1);
  mat_fill_rand(W2->val, -bound2, bound2);

  model_var *b0 =
      mv_create(arena, model, 128, 1, MY_FLAG_REQUIRES_GRAD | MY_FLAG_PARAMETER);
  model_var *b1 =
      mv_create(arena, model, 128, 1, MY_FLAG_REQUIRES_GRAD | MY_FLAG_PARAMETER);
  model_var *b2 =
      mv_create(arena, model, 10, 1, MY_FLAG_REQUIRES_GRAD | MY_FLAG_PARAMETER);

  model_var *z0_a = mv_matmul(arena, model, W0, input, 0);
  model_var *z0_b = mv_add(arena, model, z0_a, b0, 0);
  model_var *a0 = mv_relu(arena, model, z0_b, 0);

  model_var *z1_a = mv_matmul(arena, model, W1, a0, 0);
  model_var *z1_b = mv_add(arena, model, z1_a, b1, 0);
  model_var *z1_c = mv_relu(arena, model, z1_b, 0);
  model_var *a1 = mv_add(arena, model, z1_c, a0, 0);

  model_var *z2_a = mv_matmul(arena, model, W2, a1, 0);
  model_var *z2_b = mv_add(arena, model, z2_a, b2, 0);
  model_var *output = mv_softmax(arena, model, z2_b, MY_FLAG_OUTPUT);

  model_var *y = mv_create(arena, model, 10, 1, MY_FLAG_DESIRED_OUTPUT);
  mv_cross_entopy(arena, model, y, output, MY_FLAG_COST);
}

} // namespace gradientcore
