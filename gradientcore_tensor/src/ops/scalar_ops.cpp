#include "../../include/gradientcore/ops/ops.hpp"

namespace gradientcore {

Node *node_mul_scalar(Arena *arena, GraphContext *ctx, Node *a, float scalar) {
  uint32_t flags = (a->flags & NODE_FLAG_REQUIRES_GRAD) ? NODE_FLAG_REQUIRES_GRAD
                                                         : NODE_FLAG_NONE;
  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  if (out == nullptr)
    return nullptr;
  out->op = OP_MUL_SCALAR;
  out->inputs[0] = a;
  out->inputs[1] = nullptr;
  out->param = scalar;

  Tensor *tA = a->val;
  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < tA->size; i++) {
    uint64_t idx_a = tensor_get_flat_index(tA, indices);
    uint64_t idx_o = tensor_get_flat_index(out->val, indices);

    out->val->storage->data[idx_o] = tA->storage->data[idx_a] * scalar;

    for (int32_t d = tA->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < tA->shape[d])
        break;
      indices[d] = 0;
    }
  }

  return out;
}

Node *node_add_scalar(Arena *arena, GraphContext *ctx, Node *a, float scalar) {
  uint32_t flags = (a->flags & NODE_FLAG_REQUIRES_GRAD) ? NODE_FLAG_REQUIRES_GRAD
                                                         : NODE_FLAG_NONE;
  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  if (out == nullptr)
    return nullptr;
  out->op = OP_ADD_SCALAR;
  out->inputs[0] = a;
  out->inputs[1] = nullptr;
  out->param = scalar;

  Tensor *tA = a->val;
  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < tA->size; i++) {
    uint64_t idx_a = tensor_get_flat_index(tA, indices);
    uint64_t idx_o = tensor_get_flat_index(out->val, indices);

    out->val->storage->data[idx_o] = tA->storage->data[idx_a] + scalar;

    for (int32_t d = tA->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < tA->shape[d])
        break;
      indices[d] = 0;
    }
  }

  return out;
}

} // namespace gradientcore
