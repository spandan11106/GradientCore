#include "../../include/gradientcore/core/tensor.hpp"
#include <cstring>

namespace gradientcore {

Tensor *tensor_create(Arena *arena, uint32_t ndims, const uint32_t *shape) {
  if (arena == nullptr || ndims == 0 || ndims > MAX_TENSOR_DIMS)
    return nullptr;

  Tensor *t = arena->push<Tensor>();
  t->ndims = ndims;
  t->size = 1;
  t->offset = 0;

  for (uint32_t i = 0; i < ndims; i++) {
    t->shape[i] = shape[i];
    t->size *= shape[i];
  }

  t->strides[ndims - 1] = 1;
  for (int32_t i = ndims - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
  }

  t->storage = arena->push<TensorStorage>();
  t->storage->size = t->size;
  t->storage->data = arena->push_array<float>(t->size, false);

  return t;
}

Tensor *tensor_create_zeros(Arena *arena, uint32_t ndims,
                            const uint32_t *shape) {
  Tensor *t = tensor_create(arena, ndims, shape);
  if (t)
    tensor_clear(t);
  return t;
}

Tensor *tensor_view(Arena *arena, const Tensor *src) {
  Tensor *t = arena->push<Tensor>();
  t->ndims = src->ndims;
  t->size = src->size;
  t->offset = src->offset;
  t->storage = src->storage;

  std::memcpy(t->shape, src->shape, sizeof(uint32_t) * MAX_TENSOR_DIMS);
  std::memcpy(t->strides, src->strides, sizeof(uint32_t) * MAX_TENSOR_DIMS);

  return t;
}

Tensor *tensor_reshape(Arena *arena, const Tensor *src, uint32_t ndims,
                       const uint32_t *shape) {
  if (!tensor_is_contiguous(src))
    return nullptr; // Cannot easily reshape a sliced/transposed view

  uint64_t new_size = 1;
  for (uint32_t i = 0; i < ndims; i++)
    new_size *= shape[i];
  if (new_size != src->size)
    return nullptr; // Total elements must match

  Tensor *t = tensor_view(arena, src);
  t->ndims = ndims;

  for (uint32_t i = 0; i < ndims; i++)
    t->shape[i] = shape[i];

  t->strides[ndims - 1] = 1;
  for (int32_t i = ndims - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
  }

  return t;
}

Tensor *tensor_transpose(Arena *arena, const Tensor *src, uint32_t dim0,
                         uint32_t dim1) {
  if (dim0 >= src->ndims || dim1 >= src->ndims)
    return nullptr;

  Tensor *t = tensor_view(arena, src);

  uint32_t temp_shape = t->shape[dim0];
  t->shape[dim0] = t->shape[dim1];
  t->shape[dim1] = temp_shape;

  uint32_t temp_stride = t->strides[dim0];
  t->strides[dim0] = t->strides[dim1];
  t->strides[dim1] = temp_stride;

  return t;
}

uint64_t tensor_get_flat_index(const Tensor *t, const uint32_t *indices) {
  uint64_t flat_index = t->offset;
  for (uint32_t i = 0; i < t->ndims; i++) {
    flat_index += indices[i] * t->strides[i];
  }
  return flat_index;
}

bool tensor_is_contiguous(const Tensor *t) {
  uint64_t expected_stride = 1;
  for (int32_t i = t->ndims - 1; i >= 0; i--) {
    if (t->shape[i] == 1)
      continue; // Broadcasting dimensions don't break contiguity
    if (t->strides[i] != expected_stride)
      return false;
    expected_stride *= t->shape[i];
  }
  return true;
}

void tensor_clear(Tensor *t) {
  if (tensor_is_contiguous(t)) {
    std::memset(t->storage->data + t->offset, 0, t->size * sizeof(float));
  } else {
    tensor_fill(t, 0.0f);
  }
}

void tensor_fill(Tensor *t, float val) {
  if (tensor_is_contiguous(t)) {
    for (uint64_t i = 0; i < t->size; i++) {
      t->storage->data[t->offset + i] = val;
    }
  } else {
    uint32_t indices[MAX_TENSOR_DIMS] = {0};

    for (uint64_t i = 0; i < t->size; i++) {
      uint64_t flat_idx = tensor_get_flat_index(t, indices);
      t->storage->data[flat_idx] = val;

      for (int32_t d = t->ndims - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < t->shape[d]) {
          break;
        }
        indices[d] = 0;
      }
    }
  }
}

} // namespace gradientcore
