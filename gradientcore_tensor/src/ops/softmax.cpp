#include "../../include/gradientcore/ops/ops.hpp"
#include <algorithm>
#include <cmath>

namespace gradientcore {

Node *node_softmax(Arena *arena, GraphContext *ctx, Node *a) {
  uint32_t flags = NODE_FLAG_NONE;
  if (a->flags & NODE_FLAG_REQUIRES_GRAD)
    flags |= NODE_FLAG_REQUIRES_GRAD;

  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  if (out == nullptr) return nullptr;
  out->op = OP_SOFTMAX;
  out->inputs[0] = a;
  out->inputs[1] = nullptr;

  Tensor *tA = a->val;
  uint32_t C = tA->shape[tA->ndims - 1];
  uint64_t outer_size = tA->size / C;

  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < outer_size; i++) {

    indices[tA->ndims - 1] = 0;
    float max_val = tA->storage->data[tensor_get_flat_index(tA, indices)];
    
    for (uint32_t j = 1; j < C; j++) {
      indices[tA->ndims - 1] = j;
      max_val = std::max(max_val, tA->storage->data[tensor_get_flat_index(tA, indices)]);
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < C; j++) {
      indices[tA->ndims - 1] = j;
      float e = std::exp(tA->storage->data[tensor_get_flat_index(tA, indices)] - max_val);
      out->val->storage->data[tensor_get_flat_index(out->val, indices)] = e;
      sum += e;
    }

    // Avoid division by zero - if sum is 0 or extremely small, use uniform distribution
    float inv_sum = (sum > 1e-12f) ? (1.0f / sum) : (1.0f / C);
    for (uint32_t j = 0; j < C; j++) {
      indices[tA->ndims - 1] = j;
      out->val->storage->data[tensor_get_flat_index(out->val, indices)] *= inv_sum;
    }

    for (int32_t d = tA->ndims - 2; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < tA->shape[d]) break;
      indices[d] = 0;
    }
  }

  return out;
}

} // namespace gradientcore