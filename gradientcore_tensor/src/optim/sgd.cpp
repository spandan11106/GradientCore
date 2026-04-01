#include "../../include/gradientcore/optim/optim.hpp"
#include <cmath>

namespace gradientcore {
namespace optim {

SGD sgd_create(Arena *arena, Node **params, uint32_t num_params, float lr,
               float momentum) {
  SGD opt = {};
  opt.params = params;
  opt.num_params = num_params;
  opt.lr = lr;
  opt.momentum = momentum;
  opt.velocity = nullptr;

  if (momentum != 0.0f) {
    opt.velocity = arena->push_array<Tensor *>(num_params);
    for (uint32_t i = 0; i < num_params; i++) {
      Tensor *grad = params[i]->grad;
      if (grad != nullptr) {
        opt.velocity[i] = tensor_create_zeros(arena, grad->ndims, grad->shape);
      } else {
        opt.velocity[i] = nullptr;
      }
    }
  }

  return opt;
}

void sgd_step(SGD *opt) {
  for (uint32_t i = 0; i < opt->num_params; i++) {
    Node *param = opt->params[i];
    if (param->grad == nullptr || param->val == nullptr)
      continue;

    Tensor *w = param->val;
    Tensor *g = param->grad;

    uint32_t indices[MAX_TENSOR_DIMS] = {0};

    if (opt->momentum != 0.0f && opt->velocity != nullptr &&
        opt->velocity[i] != nullptr) {
      // SGD with momentum: v = momentum * v + grad
      //                    w = w - lr * v
      Tensor *vel = opt->velocity[i];
      for (uint64_t j = 0; j < w->size; j++) {
        uint64_t idx_w = tensor_get_flat_index(w, indices);
        uint64_t idx_g = tensor_get_flat_index(g, indices);
        uint64_t idx_v = tensor_get_flat_index(vel, indices);

        vel->storage->data[idx_v] =
            opt->momentum * vel->storage->data[idx_v] +
            g->storage->data[idx_g];

        w->storage->data[idx_w] -= opt->lr * vel->storage->data[idx_v];

        for (int32_t d = w->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < w->shape[d])
            break;
          indices[d] = 0;
        }
      }
    } else {
      // Vanilla SGD: w = w - lr * grad
      for (uint64_t j = 0; j < w->size; j++) {
        uint64_t idx_w = tensor_get_flat_index(w, indices);
        uint64_t idx_g = tensor_get_flat_index(g, indices);

        w->storage->data[idx_w] -= opt->lr * g->storage->data[idx_g];

        for (int32_t d = w->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < w->shape[d])
            break;
          indices[d] = 0;
        }
      }
    }
  }
}

} // namespace optim
} // namespace gradientcore
