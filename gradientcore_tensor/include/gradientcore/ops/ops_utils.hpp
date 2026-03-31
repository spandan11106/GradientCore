#pragma once
#include "../core/tensor.hpp"

namespace gradientcore {

bool broadcast_shapes(const Tensor *a, const Tensor *b, uint32_t *out_ndims,
                      uint32_t *out_shape);

Tensor *tensor_broadcast_view(Arena *arena, const Tensor *src,
                              uint32_t target_ndims,
                              const uint32_t *target_shape);

typedef float (*BinaryMathOp)(float, float);
void apply_binary_op(Tensor *out, const Tensor *a_view, const Tensor *b_view,
                     BinaryMathOp op);

} // namespace gradientcore
