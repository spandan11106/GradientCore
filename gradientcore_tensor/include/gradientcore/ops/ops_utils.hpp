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

typedef float (*UnaryMathOp)(float);
void apply_unary_op(Tensor *out, const Tensor *a, UnaryMathOp op);

void tensor_accumulate_grad(Tensor *target_grad, const Tensor *upstream_grad,
                            float scale);

void tensor_accumulate_grad_matmul(Tensor *target_grad, const Tensor *a_tensor,
                                   const Tensor *b_tensor);

void tensor_accumulate_grad_softmax(Tensor *a_grad, const Tensor *out_val,
                                    const Tensor *upstream_grad);

template <typename Func>
inline void tensor_accumulate_grad_custom(Tensor *target_grad,
                                          const Tensor *val_tensor,
                                          const Tensor *upstream_grad,
                                          Func derivative_func) {
  uint32_t indices[MAX_TENSOR_DIMS] = {0};
  for (uint64_t i = 0; i < upstream_grad->size; i++) {
    uint64_t idx_up = tensor_get_flat_index(upstream_grad, indices);
    uint64_t idx_val = tensor_get_flat_index(val_tensor, indices);

    uint32_t target_indices[MAX_TENSOR_DIMS] = {0};
    int32_t dim_offset = upstream_grad->ndims - target_grad->ndims;
    for (uint32_t d = 0; d < target_grad->ndims; d++) {
      target_indices[d] =
          (target_grad->shape[d] == 1) ? 0 : indices[d + dim_offset];
    }

    uint64_t idx_target = tensor_get_flat_index(target_grad, target_indices);

    // Apply the chain rule inline!
    target_grad->storage->data[idx_target] +=
        derivative_func(val_tensor->storage->data[idx_val],
                        upstream_grad->storage->data[idx_up]);

    for (int32_t d = upstream_grad->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < upstream_grad->shape[d])
        break;
      indices[d] = 0;
    }
  }
}

template <typename Func>
inline void tensor_accumulate_grad_binary_custom(Tensor *target_grad,
                                                 const Tensor *a_val,
                                                 const Tensor *b_val,
                                                 const Tensor *upstream_grad,
                                                 Func derivative_func) {
  uint32_t indices[MAX_TENSOR_DIMS] = {0};
  for (uint64_t i = 0; i < upstream_grad->size; i++) {
    uint64_t idx_up = tensor_get_flat_index(upstream_grad, indices);
    uint64_t idx_a = tensor_get_flat_index(a_val, indices);
    uint64_t idx_b = tensor_get_flat_index(b_val, indices);

    uint32_t target_indices[MAX_TENSOR_DIMS] = {0};
    int32_t dim_offset = upstream_grad->ndims - target_grad->ndims;
    for (uint32_t d = 0; d < target_grad->ndims; d++) {
      target_indices[d] =
          (target_grad->shape[d] == 1) ? 0 : indices[d + dim_offset];
    }

    uint64_t idx_target = tensor_get_flat_index(target_grad, target_indices);

    target_grad->storage->data[idx_target] += derivative_func(
        a_val->storage->data[idx_a], b_val->storage->data[idx_b],
        upstream_grad->storage->data[idx_up]);

    for (int32_t d = upstream_grad->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < upstream_grad->shape[d])
        break;
      indices[d] = 0;
    }
  }
}

} // namespace gradientcore
