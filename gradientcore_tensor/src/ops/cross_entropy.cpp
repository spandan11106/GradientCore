#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <cmath>

namespace gradientcore {

static float math_cross_entropy(float p, float q) {
  float qi = q > 0.0f ? q : 1e-12f;
  return -p * std::log(qi);
}

Node *node_cross_entropy(Arena *arena, GraphContext *ctx, Node *p, Node *q) {
  uint32_t out_ndims;
  uint32_t out_shape[MAX_TENSOR_DIMS];

  if (!broadcast_shapes(p->val, q->val, &out_ndims, out_shape))
    return nullptr;

  Tensor *p_bcast = tensor_broadcast_view(arena, p->val, out_ndims, out_shape);
  Tensor *q_bcast = tensor_broadcast_view(arena, q->val, out_ndims, out_shape);

  uint32_t flags = NODE_FLAG_NONE;
  if ((p->flags & NODE_FLAG_REQUIRES_GRAD) ||
      (q->flags & NODE_FLAG_REQUIRES_GRAD)) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }

  Node *out = node_create(arena, ctx, out_ndims, out_shape, flags);
  if (out == nullptr) return nullptr;
  out->op = OP_CROSS_ENTROPY;
  out->inputs[0] = p;
  out->inputs[1] = q;

  apply_binary_op(out->val, p_bcast, q_bcast, math_cross_entropy);
  return out;
}

} // namespace gradientcore
