#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"

namespace gradientcore {

Node *node_matmul(Arena *arena, GraphContext *ctx, Node *a, Node *b) {
  Tensor *tA = a->val;
  Tensor *tB = b->val;

  if (tA->ndims < 2 || tB->ndims < 2)
    return nullptr;

  uint32_t M = tA->shape[tA->ndims - 2];
  uint32_t K = tA->shape[tA->ndims - 1];
  uint32_t K_b = tB->shape[tB->ndims - 2];
  uint32_t N = tB->shape[tB->ndims - 1];

  if (K != K_b)
    return nullptr;

  uint32_t out_ndims = tA->ndims;
  uint32_t out_shape[MAX_TENSOR_DIMS];
  for (uint32_t i = 0; i < out_ndims - 2; i++) {
    out_shape[i] = tA->shape[i];
  }
  out_shape[out_ndims - 2] = M;
  out_shape[out_ndims - 1] = N;

  uint32_t flags = NODE_FLAG_NONE;
  if ((a->flags & NODE_FLAG_REQUIRES_GRAD) ||
      (b->flags & NODE_FLAG_REQUIRES_GRAD)) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }

  Node *out = node_create(arena, ctx, out_ndims, out_shape, flags);
  out->op = OP_MATMUL;
  out->inputs[0] = a;
  out->inputs[1] = b;

  uint64_t batch_count = 1;
  for (uint32_t i = 0; i < out_ndims - 2; i++) {
    batch_count *= out_shape[i];
  }

  uint32_t batch_indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t b_idx = 0; b_idx < batch_count; b_idx++) {
    for (uint32_t i = 0; i < M; i++) {
      for (uint32_t j = 0; j < N; j++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
          uint32_t a_idx[MAX_TENSOR_DIMS] = {0};
          uint32_t b_idx_arr[MAX_TENSOR_DIMS] = {0};

          for (uint32_t d = 0; d < out_ndims - 2; d++) {
            a_idx[d] = batch_indices[d];
            b_idx_arr[d] = batch_indices[d];
          }

          a_idx[tA->ndims - 2] = i;
          a_idx[tA->ndims - 1] = k;
          b_idx_arr[tB->ndims - 2] = k;
          b_idx_arr[tB->ndims - 1] = j;

          float valA = tA->storage->data[tensor_get_flat_index(tA, a_idx)];
          float valB = tB->storage->data[tensor_get_flat_index(tB, b_idx_arr)];
          sum += valA * valB;
        }

        uint32_t c_idx[MAX_TENSOR_DIMS] = {0};
        for (uint32_t d = 0; d < out_ndims - 2; d++)
          c_idx[d] = batch_indices[d];
        c_idx[out->val->ndims - 2] = i;
        c_idx[out->val->ndims - 1] = j;

        out->val->storage->data[tensor_get_flat_index(out->val, c_idx)] = sum;
      }
    }

    for (int32_t d = out_ndims - 3; d >= 0; d--) {
      batch_indices[d]++;
      if (batch_indices[d] < out->val->shape[d])
        break;
      batch_indices[d] = 0;
    }
  }

  return out;
}

} // namespace gradientcore

