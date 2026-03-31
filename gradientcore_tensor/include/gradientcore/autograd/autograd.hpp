#pragma once

#include "../core/arena.hpp"
#include "../core/tensor.hpp"
#include <cstdint>

namespace gradientcore {

enum NodeFlags {
  NODE_FLAG_NONE = 0,
  NODE_FLAG_REQUIRES_GRAD = (1 << 0),
  NODE_FLAG_PARAMETER = (1 << 1),
  // We drop INPUT/OUTPUT flags from here as they are more
  // relevant to higher level nn::Module abstractions.
};

enum OpType {
  OP_NULL,
  OP_CREATE,

  // Unary Ops
  _OP_UNARY_START,
  OP_RELU,
  OP_SOFTMAX,
  OP_SIGMOID,
  OP_TANH,
  OP_GELU,
  OP_SILU,
  OP_LEAKY_RELU,

  // Binary Ops
  _OP_BINARY_START,
  OP_ADD,
  OP_SUB,
  OP_MUL, // Element-wise multiplications
  OP_MATMUL,
  OP_CROSS_ENTROPY,
  OP_MSE,
  OP_L1_LOSS,
  OP_BCE,
};

#define MAX_NODE_INPUTS 2
#define NUM_INPUTS(op)                                                         \
  ((op) < _OP_UNARY_START ? 0 : ((op) < _OP_BINARY_START ? 1 : 2))

struct Node {
  uint32_t index;
  uint32_t flags;

  Tensor *val;
  Tensor *grad;

  OpType op;
  Node *inputs[MAX_NODE_INPUTS];
};

struct GraphContext {
  uint32_t num_nodes;
};

GraphContext *graph_create(Arena *arena);

Node *node_create(Arena *arena, GraphContext *ctx, uint32_t ndims,
                  const uint32_t *shape, uint32_t flags);

} // namespace gradientcore
