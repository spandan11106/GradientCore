#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"

namespace gradientcore {

static float math_div(float a, float b) { return a / b; }

Node *node_div(Arena *arena, GraphContext *ctx, Node *a, Node *b) {
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
  if (out == nullptr)
    return nullptr;
  out->op = OP_DIV;
  out->inputs[0] = a;
  out->inputs[1] = b;

  apply_binary_op(out->val, a_bcast, b_bcast, math_div);
  return out;
}

} // namespace gradientcore
