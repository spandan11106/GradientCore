#pragma once

#include "../autograd/autograd.hpp"
#include "../core/tensor.hpp"

namespace gradientcore {
namespace optim {

struct SGD {
  Node **params;
  uint32_t num_params;
  float lr;
  float momentum;

  // Per-parameter momentum velocity buffers (nullptr if momentum == 0)
  Tensor **velocity;
};

SGD sgd_create(Arena *arena, Node **params, uint32_t num_params, float lr,
               float momentum = 0.0f);

void sgd_step(SGD *opt);

struct Adam {
  Node **params;
  uint32_t num_params;
  float lr;
  float beta1;
  float beta2;
  float epsilon;

  // Per-parameter state
  Tensor **m; // First moment estimate
  Tensor **v; // Second moment estimate
  uint32_t t; // Timestep counter
};

Adam adam_create(Arena *arena, Node **params, uint32_t num_params,
                float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
                float epsilon = 1e-8f);

void adam_step(Adam *opt);

// Shared utility: zeros all parameter gradients
void zero_grad(Node **params, uint32_t num_params);

} // namespace optim
} // namespace gradientcore
