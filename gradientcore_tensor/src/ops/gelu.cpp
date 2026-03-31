#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <cmath>

namespace gradientcore {

static float math_gelu(float x) {
  const float SQRT_2_OVER_PI = 0.7978845608f;
  return 0.5f * x *
         (1.0f + std::tanh(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
}

Node *node_gelu(Arena *arena, GraphContext *ctx, Node *a) {
  uint32_t flags = (a->flags & NODE_FLAG_REQUIRES_GRAD)
                       ? NODE_FLAG_REQUIRES_GRAD
                       : NODE_FLAG_NONE;
  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  out->op = OP_GELU;
  out->inputs[0] = a;

  apply_unary_op(out->val, a->val, math_gelu);
  return out;
}

} // namespace gradientcore
