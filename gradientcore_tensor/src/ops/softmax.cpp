#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <algorithm>
#include <cmath>

namespace gradientcore {

Node *node_softmax(Arena *arena, GraphContext *ctx, Node *a) {
  uint32_t flags = NODE_FLAG_NONE;
  if (a->flags & NODE_FLAG_REQUIRES_GRAD)
    flags |= NODE_FLAG_REQUIRES_GRAD;

  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  out->op = OP_SOFTMAX;
  out->inputs[0] = a;

  Tensor *tA = a->val;
  uint32_t C = tA->shape[tA->ndims - 1];
  uint64_t outer_size = tA->size / C;

  for (uint64_t i = 0; i < outer_size; i++) {
    uint64_t offset = i * C;

    float max_val = tA->storage->data[offset];
    for (uint32_t j = 1; j < C; j++) {
      max_val = std::max(max_val, tA->storage->data[offset + j]);
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < C; j++) {
      float e = std::exp(tA->storage->data[offset + j] - max_val);
      out->val->storage->data[offset + j] = e;
      sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (uint32_t j = 0; j < C; j++) {
      out->val->storage->data[offset + j] *= inv_sum;
    }
  }

  return out;
}

} // namespace gradientcore
