#include "../../include/gradientcore/core/tensor.hpp"
#include <cstdint>
#include <cstring>

namespace gradientcore {

Tensor *tensor_create(Arena *arena, uint32_t ndims, const uint32_t *shape) {
  if (arena == nullptr || ndims == 0 || ndims > MAX_TENSOR_DIMS) {
    return nullptr;
  }

  Tensor *t = arena->push<Tensor>();
  t->ndims = ndims;
  t->size = 1;

  for (uint32_t i = 0; i < ndims; i++) {
    t->shape[i] = shape[i];
    t->size *= shape[i];
  }

  t->strides[ndims - 1] = 1;
  for (int32_t i = ndims - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
  }

  t->data = arena->push_array<float>(t->size, false);

  return t;
}

Tensor *tensor_create_zeros(Arena *arena, uint32_t ndims,
                            const uint32_t *shape) {
  Tensor *t = tensor_create(arena, ndims, shape);
  if (t) {
    tensor_clear(t);
  }
  return t;
}

uint64_t tensor_get_flat_index(const Tensor *t, const uint32_t *indices) {
  uint64_t flat_index = 0;
  for (uint32_t i = 0; i < t->ndims; i++) {
    flat_index += indices[i] * t->strides[i];
  }
  return flat_index;
}

void tensor_clear(Tensor *t) {
  std::memset(t->data, 0, t->size * sizeof(float));
}

bool tensor_copy(Tensor *dst, const Tensor *src) {
  if (dst->size != src->size)
    return false;

  std::memcpy(dst->data, src->data, dst->size * sizeof(float));
  return true;
}

void tensor_fill(Tensor *t, float val) {
  for (uint64_t i = 0; i < t->size; i++) {
    t->data[i] = val;
  }
}

} // namespace gradientcore
