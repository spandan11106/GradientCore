#include "../../include/gradientcore/ops/ops.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <cmath>

namespace gradientcore {

static float math_bce(float pred, float target) {
  // Clamp predictions to avoid log(0)
  float p = std::fmax(std::fmin(pred, 1.0f - 1e-12f), 1e-12f);
  return -(target * std::log(p) + (1.0f - target) * std::log(1.0f - p));
}

Node *node_bce(Arena *arena, GraphContext *ctx, Node *pred, Node *target) {
  uint32_t out_ndims;
  uint32_t out_shape[MAX_TENSOR_DIMS];
  if (!broadcast_shapes(pred->val, target->val, &out_ndims, out_shape))
    return nullptr;

  Tensor *p_bcast =
      tensor_broadcast_view(arena, pred->val, out_ndims, out_shape);
  Tensor *t_bcast =
      tensor_broadcast_view(arena, target->val, out_ndims, out_shape);

  uint32_t flags = ((pred->flags | target->flags) & NODE_FLAG_REQUIRES_GRAD)
                       ? NODE_FLAG_REQUIRES_GRAD
                       : NODE_FLAG_NONE;
  Node *out = node_create(arena, ctx, out_ndims, out_shape, flags);
  if (out == nullptr) return nullptr;
  out->op = OP_BCE;
  out->inputs[0] = pred;
  out->inputs[1] = target;

  apply_binary_op(out->val, p_bcast, t_bcast, math_bce);
  return out;
}

} // namespace gradientcore
