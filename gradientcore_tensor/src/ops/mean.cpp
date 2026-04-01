#include "../../include/gradientcore/ops/ops.hpp"

namespace gradientcore {

Node *node_mean(Arena *arena, GraphContext *ctx, Node *a) {
  uint32_t flags = (a->flags & NODE_FLAG_REQUIRES_GRAD) ? NODE_FLAG_REQUIRES_GRAD
                                                         : NODE_FLAG_NONE;
  uint32_t out_shape[1] = {1};
  Node *out = node_create(arena, ctx, 1, out_shape, flags);
  if (out == nullptr)
    return nullptr;
  out->op = OP_MEAN;
  out->inputs[0] = a;
  out->inputs[1] = nullptr;

  Tensor *tA = a->val;
  float sum = 0.0f;
  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < tA->size; i++) {
    uint64_t idx = tensor_get_flat_index(tA, indices);
    sum += tA->storage->data[idx];

    for (int32_t d = tA->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < tA->shape[d])
        break;
      indices[d] = 0;
    }
  }

  out->val->storage->data[out->val->offset] = sum / (float)tA->size;
  return out;
}

} // namespace gradientcore
