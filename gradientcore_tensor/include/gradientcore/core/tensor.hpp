#pragma once

#include "arena.hpp"
#include <cstdint>

namespace gradientcore {

constexpr uint32_t MAX_TENSOR_DIMS = 10;

// The actual chunk of memory.
struct TensorStorage {
  float *data;
  uint64_t size;
};

// A "View" into a TensorStorage.
struct Tensor {
  uint32_t ndims;
  uint32_t shape[MAX_TENSOR_DIMS];
  uint32_t strides[MAX_TENSOR_DIMS];
  uint64_t size;   // Number of elements in this specific view
  uint64_t offset; // Starting index in the storage->data array

  TensorStorage *storage; // Pointer to the underlying memory
};

// Memory allocation
Tensor *tensor_create(Arena *arena, uint32_t ndims, const uint32_t *shape);
Tensor *tensor_create_zeros(Arena *arena, uint32_t ndims,
                            const uint32_t *shape);

Tensor *tensor_view(Arena *arena, const Tensor *src);
Tensor *tensor_reshape(Arena *arena, const Tensor *src, uint32_t ndims,
                       const uint32_t *shape);
Tensor *tensor_transpose(Arena *arena, const Tensor *src, uint32_t dim0,
                         uint32_t dim1);

// Utility functions
uint64_t tensor_get_flat_index(const Tensor *t, const uint32_t *indices);
void tensor_clear(Tensor *t);
bool tensor_copy(Tensor *dst, const Tensor *src);
void tensor_fill(Tensor *t, float val);
bool tensor_is_contiguous(const Tensor *t);

} // namespace gradientcore
