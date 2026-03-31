#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"

namespace gradientcore {

static float math_mse(float a, float b) {
  float diff = a - b;
  return diff * diff;
}

Node *node_mse(Arena *arena, GraphContext *ctx, Node *a, Node *b) {
  uint32_t out_ndims;
  uint32_t out_shape[MAX_TENSOR_DIMS];

  if (!broadcast_shapes(a->val, b->val, &out_ndims, out_shape))
    return nullptr;

  Tensor *a_bcast = tensor_broadcast_view(arena, a->val, out_ndims, out_shape);
  Tensor *b_bcast = tensor_broadcast_view(arena, b->val, out_ndims, out_shape);

  uint32_t flags = NODE_FLAG_NONE;
  if ((a->flags & NODE_FLAG_REQUIRES_GRAD) ||
      (b->flags & NODE_FLAG_REQUIRES_GRAD)) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }

  Node *out = node_create(arena, ctx, out_ndims, out_shape, flags);
  out->op = OP_MSE;
  out->inputs[0] = a;
  out->inputs[1] = b;

  apply_binary_op(out->val, a_bcast, b_bcast, math_mse);
  return out;
}

} // namespace gradientcore
