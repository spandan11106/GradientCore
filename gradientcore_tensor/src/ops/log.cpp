#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <cmath>

namespace gradientcore {

static float math_log(float a) { return std::log(a); }

Node *node_log(Arena *arena, GraphContext *ctx, Node *a) {
  uint32_t flags = (a->flags & NODE_FLAG_REQUIRES_GRAD) ? NODE_FLAG_REQUIRES_GRAD
                                                         : NODE_FLAG_NONE;
  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  if (out == nullptr)
    return nullptr;
  out->op = OP_LOG;
  out->inputs[0] = a;
  out->inputs[1] = nullptr;

  apply_unary_op(out->val, a->val, math_log);
  return out;
}

} // namespace gradientcore
