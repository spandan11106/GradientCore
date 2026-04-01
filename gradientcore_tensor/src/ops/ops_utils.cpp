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

    a_idx--; b_idx--; out_idx--;
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
      if (indices[d] < out->shape[d]) break;
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
      if (indices[d] < out->shape[d]) break;
      indices[d] = 0;
    }
  }
}

void tensor_accumulate_grad(Tensor *target_grad, const Tensor *upstream_grad,
                            float scale) {
  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  // upstream_grad->ndims should always be >= target_grad->ndims for
  // broadcasting. If not, the operation is invalid.
  if (upstream_grad->ndims < target_grad->ndims)
    return;

  uint32_t dim_offset = upstream_grad->ndims - target_grad->ndims;

  for (uint64_t i = 0; i < upstream_grad->size; i++) {
    uint64_t idx_upstream = tensor_get_flat_index(upstream_grad, indices);

    uint32_t target_indices[MAX_TENSOR_DIMS] = {0};

    for (uint32_t d = 0; d < target_grad->ndims; d++) {
      uint32_t up_d = d + dim_offset;
      target_indices[d] = (target_grad->shape[d] == 1) ? 0 : indices[up_d];
    }

    uint64_t idx_target = tensor_get_flat_index(target_grad, target_indices);

    target_grad->storage->data[idx_target] +=
        upstream_grad->storage->data[idx_upstream] * scale;

    for (int32_t d = upstream_grad->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < upstream_grad->shape[d]) break;
      indices[d] = 0;
    }
  }
}

void tensor_accumulate_grad_matmul(Tensor *target_grad, const Tensor *tA,
                                   const Tensor *tB) {
  if (tA->ndims < 2 || tB->ndims < 2) return;

  uint32_t M = tA->shape[tA->ndims - 2];
  uint32_t K = tA->shape[tA->ndims - 1];
  uint32_t N = tB->shape[tB->ndims - 1];

  uint32_t out_ndims = std::max({target_grad->ndims, tA->ndims, tB->ndims});
  
  uint32_t offsetT = out_ndims - target_grad->ndims;
  uint32_t offsetA = out_ndims - tA->ndims;
  uint32_t offsetB = out_ndims - tB->ndims;

  uint32_t batch_shape[MAX_TENSOR_DIMS];
  uint64_t batch_count = 1;
  
  for (uint32_t d = 0; d < out_ndims - 2; d++) {
    uint32_t dimT = (d >= offsetT) ? target_grad->shape[d - offsetT] : 1;
    uint32_t dimA = (d >= offsetA) ? tA->shape[d - offsetA] : 1;
    uint32_t dimB = (d >= offsetB) ? tB->shape[d - offsetB] : 1;
    batch_shape[d] = std::max({dimT, dimA, dimB});
    batch_count *= batch_shape[d];
  }

  uint32_t batch_indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t b_idx = 0; b_idx < batch_count; b_idx++) {
    for (uint32_t i = 0; i < M; i++) {
      for (uint32_t j = 0; j < N; j++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
          uint32_t a_idx[MAX_TENSOR_DIMS] = {0};
          uint32_t b_idx_arr[MAX_TENSOR_DIMS] = {0};

          for(uint32_t d = 0; d < out_ndims - 2; d++){
              if (d >= offsetA) a_idx[d - offsetA] = (tA->shape[d - offsetA] == 1) ? 0 : batch_indices[d];
              if (d >= offsetB) b_idx_arr[d - offsetB] = (tB->shape[d - offsetB] == 1) ? 0 : batch_indices[d];
          }

          a_idx[tA->ndims - 2] = i; a_idx[tA->ndims - 1] = k;
          b_idx_arr[tB->ndims - 2] = k; b_idx_arr[tB->ndims - 1] = j;

          float valA = tA->storage->data[tensor_get_flat_index(tA, a_idx)];
          float valB = tB->storage->data[tensor_get_flat_index(tB, b_idx_arr)];
          sum += valA * valB;
        }

        uint32_t c_idx[MAX_TENSOR_DIMS] = {0};
        for(uint32_t d = 0; d < out_ndims - 2; d++) {
            if (d >= offsetT) c_idx[d - offsetT] = (target_grad->shape[d - offsetT] == 1) ? 0 : batch_indices[d];
        }
        c_idx[target_grad->ndims - 2] = i; c_idx[target_grad->ndims - 1] = j;
        
        target_grad->storage->data[tensor_get_flat_index(target_grad, c_idx)] += sum;
      }
    }

    if (out_ndims > 2) {
      for (int32_t d = out_ndims - 3; d >= 0; d--) {
        batch_indices[d]++;
        if (batch_indices[d] < batch_shape[d]) break;
        batch_indices[d] = 0;
      }
    }
  }
}

void tensor_accumulate_grad_softmax(Tensor *a_grad, const Tensor *out_val,
                                    const Tensor *upstream_grad) {
  uint32_t C = out_val->shape[out_val->ndims - 1];
  uint64_t outer_size = out_val->size / C;

  uint32_t batch_indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t b = 0; b < outer_size; b++) {
    float dot = 0.0f;
    for (uint32_t j = 0; j < C; j++) {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint32_t d = 0; d < out_val->ndims - 1; d++) {
        indices[d] = batch_indices[d];
      }
      indices[out_val->ndims - 1] = j;
      
      float y_i = out_val->storage->data[tensor_get_flat_index(out_val, indices)];
      float dy_i = upstream_grad->storage->data[tensor_get_flat_index(upstream_grad, indices)];
      dot += y_i * dy_i;
    }

    // Accumulate gradients
    for (uint32_t j = 0; j < C; j++) {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      // Copy batch indices
      for (uint32_t d = 0; d < out_val->ndims - 1; d++) {
        indices[d] = batch_indices[d];
      }
      indices[out_val->ndims - 1] = j;

      uint64_t idx_out = tensor_get_flat_index(out_val, indices);
      uint64_t idx_up = tensor_get_flat_index(upstream_grad, indices);
      uint64_t idx_grad = tensor_get_flat_index(a_grad, indices);

      float y_i = out_val->storage->data[idx_out];
      float dy_i = upstream_grad->storage->data[idx_up];
      a_grad->storage->data[idx_grad] += y_i * (dy_i - dot);
    }

    // Increment batch indices (odometer)
    for (int32_t d = (int32_t)out_val->ndims - 2; d >= 0; d--) {
      batch_indices[d]++;
      if (batch_indices[d] < out_val->shape[d]) break;
      batch_indices[d] = 0;
    }
  }
}

} // namespace gradientcore