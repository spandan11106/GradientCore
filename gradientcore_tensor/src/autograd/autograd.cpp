#include "../../include/gradientcore/autograd/autograd.hpp"
#include <cstdint>

namespace gradientcore {

GraphContext *graph_create(Arena *arena) {
  if (arena == nullptr)
    return nullptr;

  GraphContext *ctx = arena->push<GraphContext>();
  ctx->num_nodes = 0;

  return ctx;
}

Node *node_create(Arena *arena, GraphContext *ctx, uint32_t ndims,
                  const uint32_t *shape, uint32_t flags) {
  if (arena == nullptr || ctx == nullptr)
    return nullptr;

  Node *out = arena->push<Node>();

  out->index = ctx->num_nodes++;
  out->flags = flags;
  out->op = OP_CREATE;

  out->val = tensor_create_zeros(arena, ndims, shape);

  if (flags & NODE_FLAG_REQUIRES_GRAD) {
    out->grad = tensor_create_zeros(arena, ndims, shape);
  } else
    out->grad = nullptr;

  out->inputs[0] = nullptr;
  out->inputs[1] = nullptr;
  out->param = 0.0f;

  return out;
}

} // namespace gradientcore
