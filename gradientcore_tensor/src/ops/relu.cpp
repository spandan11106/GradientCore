#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <algorithm>

namespace gradientcore {

static float math_relu(float a) { return std::max(0.0f, a); }

Node *node_relu(Arena *arena, GraphContext *ctx, Node *a) {
  uint32_t flags = NODE_FLAG_NONE;
  if (a->flags & NODE_FLAG_REQUIRES_GRAD) {
    flags |= NODE_FLAG_REQUIRES_GRAD;
  }

  Node *out = node_create(arena, ctx, a->val->ndims, a->val->shape, flags);
  if (out == nullptr) return nullptr;
  out->op = OP_RELU;
  out->inputs[0] = a;
  out->inputs[1] = nullptr;

  apply_unary_op(out->val, a->val, math_relu);

  return out;
}

} // namespace gradientcore
