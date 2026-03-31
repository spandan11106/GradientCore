#pragma once

#include "arena.hpp"
#include <cstdint>
#include <sys/types.h>

namespace gradientcore {

constexpr uint64_t MAX_TENSOR_DIMS = 10;

struct Tensor {
  uint32_t ndims;
  uint32_t shape[MAX_TENSOR_DIMS];
  uint32_t strides[MAX_TENSOR_DIMS]; // Memory steps required to move 1 element
                                     // in each dim
  uint64_t size;
  float *data;
};

Tensor *tensor_create(Arena *arena, uint32_t ndims, const uint32_t *shape);
Tensor *tensor_create_zeros(Arena *arena, uint32_t ndims,
                            const uint32_t *shape);

uint64_t tensor_get_flat_index(const Tensor *t, const uint32_t *indices);
void tensor_clear(Tensor *t);
bool tensor_copy(Tensor *dst, const Tensor *src);
void tensor_fill(Tensor *t, float val);

} // namespace gradientcore
