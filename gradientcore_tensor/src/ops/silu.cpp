#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <cmath>

namespace gradientcore {

static float math_silu(float x) { return x / (1.0f + std::exp(-x)); }

Node *node_silu(Arena *arena, GraphContext *ctx, Node *a) {
  uint32_t flags = (a->flags & NODE_FLAG_REQUIRES_GRAD)
                       ? NODE_FLAG_REQUIRES_GRAD
                       : NODE_FLAG_NONE;
  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  out->op = OP_SILU;
  out->inputs[0] = a;

  apply_unary_op(out->val, a->val, math_silu);
  return out;
}

} // namespace gradientcore
