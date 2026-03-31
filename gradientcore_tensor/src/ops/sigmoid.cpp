#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <cmath>

namespace gradientcore {

static float math_sigmoid(float a) { return 1.0f / (1.0f + std::exp(-a)); }

Node *node_sigmoid(Arena *arena, GraphContext *ctx, Node *a) {
  uint32_t flags = NODE_FLAG_NONE;
  if (a->flags & NODE_FLAG_REQUIRES_GRAD)
    flags |= NODE_FLAG_REQUIRES_GRAD;

  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  out->op = OP_SIGMOID;
  out->inputs[0] = a;

  apply_unary_op(out->val, a->val, math_sigmoid);
  return out;
}

} // namespace gradientcore
