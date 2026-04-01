#include "../../../gradientcore/include/gradientcore/base/prng.hpp"
#include "../../include/gradientcore/nn/nn.hpp"
#include <cmath>

namespace gradientcore {
namespace nn {

void init_uniform(Tensor *t, float bound) {
  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < t->size; i++) {
    uint64_t idx = tensor_get_flat_index(t, indices);

    float r = prng::randf() * 2.0f - 1.0f;
    t->storage->data[idx] = r * bound;

    for (int32_t d = t->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < t->shape[d])
        break;
      indices[d] = 0;
    }
  }
}

void init_kaiming_uniform(Tensor *t, uint32_t fan_in) {
  float bound = std::sqrt(6.0f / (float)fan_in);
  init_uniform(t, bound);
}

void init_xavier_uniform(Tensor *t, uint32_t fan_in, uint32_t fan_out) {
  float bound = std::sqrt(6.0f / (float)(fan_in + fan_out));
  init_uniform(t, bound);
}

} // namespace nn
} // namespace gradientcore
