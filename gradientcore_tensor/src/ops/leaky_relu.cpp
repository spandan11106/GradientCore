#include "../../include/gradientcore/ops/ops.hpp"

namespace gradientcore {

Node *node_leaky_relu(Arena *arena, GraphContext *ctx, Node *a,
                      float negative_slope) {
  uint32_t flags = NODE_FLAG_NONE;
  if (a->flags & NODE_FLAG_REQUIRES_GRAD) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }

  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  if (out == nullptr) return nullptr;
  out->op = OP_LEAKY_RELU;
  out->inputs[0] = a;
  out->inputs[1] = nullptr;
  out->param = negative_slope;  

  uint32_t indices[MAX_TENSOR_DIMS] = {0};
  for (uint64_t i = 0; i < out->val->size; i++) {
    uint64_t idx_a = tensor_get_flat_index(a->val, indices);
    uint64_t idx_out = tensor_get_flat_index(out->val, indices);
    float val = a->val->storage->data[idx_a];

    out->val->storage->data[idx_out] = (val > 0.0f) ? val : val * negative_slope;

    for (int32_t d = out->val->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < out->val->shape[d])
        break;
      indices[d] = 0;
    }
  }

  return out;
}

} // namespace gradientcore
