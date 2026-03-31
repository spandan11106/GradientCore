#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <algorithm>

namespace gradientcore {

bool broadcast_shapes(const Tensor *a, const Tensor *b, uint32_t *out_ndims,
                      uint32_t *out_shape) {
  *out_ndims = std::max(a->ndims, b->ndims);

  int32_t a_idx = a->ndims - 1;
  int32_t b_idx = b->ndims - 1;
  int32_t out_idx = *out_ndims - 1;

  while (out_idx >= 0) {
    uint32_t dim_a = (a_idx >= 0) ? a->shape[a_idx] : 1;
    uint32_t dim_b = (b_idx >= 0) ? b->shape[b_idx] : 1;

    if (dim_a == dim_b) {
      out_shape[out_idx] = dim_a;
    } else if (dim_a == 1) {
      out_shape[out_idx] = dim_b;
    } else if (dim_b == 1) {
      out_shape[out_idx] = dim_a;
    } else {
      return false;
    }

    a_idx--;
    b_idx--;
    out_idx--;
  }

  return true;
}

Tensor *tensor_broadcast_view(Arena *arena, const Tensor *src,
                              uint32_t target_ndims,
                              const uint32_t *target_shape) {
  Tensor *t = tensor_view(arena, src);
  t->ndims = target_ndims;

  uint64_t new_size = 1;
  int32_t src_idx = src->ndims - 1;

  for (int32_t i = target_ndims - 1; i >= 0; i--) {
    t->shape[i] = target_shape[i];
    new_size *= target_shape[i];

    uint32_t dim_src = (src_idx >= 0) ? src->shape[src_idx] : 1;

    if (dim_src == target_shape[i]) {
      t->strides[i] = (src_idx >= 0) ? src->strides[src_idx] : 0;
    } else {
      t->strides[i] = 0;
    }
    src_idx--;
  }
  t->size = new_size;
  return t;
}

void apply_unary_op(Tensor *out, const Tensor *a, UnaryMathOp op) {
  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < out->size; i++) {
    uint64_t idx_out = tensor_get_flat_index(out, indices);
    uint64_t idx_a = tensor_get_flat_index(a, indices);

    out->storage->data[idx_out] = op(a->storage->data[idx_a]);

    for (int32_t d = out->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < out->shape[d])
        break;
      indices[d] = 0;
    }
  }
}

void apply_binary_op(Tensor *out, const Tensor *a_view, const Tensor *b_view,
                     BinaryMathOp op) {
  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < out->size; i++) {
    uint64_t idx_out = tensor_get_flat_index(out, indices);
    uint64_t idx_a = tensor_get_flat_index(a_view, indices);
    uint64_t idx_b = tensor_get_flat_index(b_view, indices);

    out->storage->data[idx_out] =
        op(a_view->storage->data[idx_a], b_view->storage->data[idx_b]);

    for (int32_t d = out->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < out->shape[d])
        break;
      indices[d] = 0;
    }
  }
}

} // namespace gradientcore
