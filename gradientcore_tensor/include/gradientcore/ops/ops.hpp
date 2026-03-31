#pragma once
#include "../autograd/autograd.hpp"

namespace gradientcore {

Node *node_add(Arena *arena, GraphContext *ctx, Node *a, Node *b);
Node *node_sub(Arena *arena, GraphContext *ctx, Node *a, Node *b);

} // namespace gradientcore
