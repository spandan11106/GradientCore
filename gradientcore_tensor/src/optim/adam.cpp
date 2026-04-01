#include "../../include/gradientcore/optim/optim.hpp"
#include <cmath>

namespace gradientcore {
namespace optim {

Adam adam_create(Arena *arena, Node **params, uint32_t num_params, float lr,
                float beta1, float beta2, float epsilon) {
  Adam opt = {};
  opt.params = params;
  opt.num_params = num_params;
  opt.lr = lr;
  opt.beta1 = beta1;
  opt.beta2 = beta2;
  opt.epsilon = epsilon;
  opt.t = 0;

  opt.m = arena->push_array<Tensor *>(num_params);
  opt.v = arena->push_array<Tensor *>(num_params);

  for (uint32_t i = 0; i < num_params; i++) {
    Tensor *grad = params[i]->grad;
    if (grad != nullptr) {
      opt.m[i] = tensor_create_zeros(arena, grad->ndims, grad->shape);
      opt.v[i] = tensor_create_zeros(arena, grad->ndims, grad->shape);
    } else {
      opt.m[i] = nullptr;
      opt.v[i] = nullptr;
    }
  }

  return opt;
}

void adam_step(Adam *opt) {
  opt->t++;

  float bc1 = 1.0f - std::pow(opt->beta1, (float)opt->t);
  float bc2 = 1.0f - std::pow(opt->beta2, (float)opt->t);

  for (uint32_t i = 0; i < opt->num_params; i++) {
    Node *param = opt->params[i];
    if (param->grad == nullptr || param->val == nullptr)
      continue;
    if (opt->m[i] == nullptr || opt->v[i] == nullptr)
      continue;

    Tensor *w = param->val;
    Tensor *g = param->grad;
    Tensor *m = opt->m[i];
    Tensor *v = opt->v[i];

    uint32_t indices[MAX_TENSOR_DIMS] = {0};

    for (uint64_t j = 0; j < w->size; j++) {
      uint64_t idx_w = tensor_get_flat_index(w, indices);
      uint64_t idx_g = tensor_get_flat_index(g, indices);
      uint64_t idx_m = tensor_get_flat_index(m, indices);
      uint64_t idx_v = tensor_get_flat_index(v, indices);

      float grad = g->storage->data[idx_g];

      // m = beta1 * m + (1 - beta1) * grad
      m->storage->data[idx_m] =
          opt->beta1 * m->storage->data[idx_m] + (1.0f - opt->beta1) * grad;

      // v = beta2 * v + (1 - beta2) * grad^2
      v->storage->data[idx_v] = opt->beta2 * v->storage->data[idx_v] +
                                 (1.0f - opt->beta2) * grad * grad;

      // Bias-corrected estimates
      float m_hat = m->storage->data[idx_m] / bc1;
      float v_hat = v->storage->data[idx_v] / bc2;

      // w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
      w->storage->data[idx_w] -=
          opt->lr * m_hat / (std::sqrt(v_hat) + opt->epsilon);

      for (int32_t d = w->ndims - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < w->shape[d])
          break;
        indices[d] = 0;
      }
    }
  }
}

} // namespace optim
} // namespace gradientcore
