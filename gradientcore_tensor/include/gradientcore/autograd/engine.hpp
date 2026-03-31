#pragma once
#include "autograd.hpp"

namespace gradientcore {

struct GraphProgram {
  Node **nodes;
  uint32_t size;
};

GraphProgram graph_compile(Arena *arena, GraphContext *ctx, Node *out_node);

void graph_backward(GraphProgram *prog);

} // namespace gradientcore
