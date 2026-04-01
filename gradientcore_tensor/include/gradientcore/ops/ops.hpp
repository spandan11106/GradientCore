#pragma once
#include "../autograd/autograd.hpp"

namespace gradientcore {

// Basic arithmetic
Node *node_add(Arena *arena, GraphContext *ctx, Node *a, Node *b);
Node *node_sub(Arena *arena, GraphContext *ctx, Node *a, Node *b);
Node *node_mul(Arena *arena, GraphContext *ctx, Node *a, Node *b);
Node *node_div(Arena *arena, GraphContext *ctx, Node *a, Node *b);
Node *node_matmul(Arena *arena, GraphContext *ctx, Node *a, Node *b);
Node *node_neg(Arena *arena, GraphContext *ctx, Node *a);

// Scalar ops (scalar stored in node->param)
Node *node_mul_scalar(Arena *arena, GraphContext *ctx, Node *a, float scalar);
Node *node_add_scalar(Arena *arena, GraphContext *ctx, Node *a, float scalar);

// Element-wise math
Node *node_exp(Arena *arena, GraphContext *ctx, Node *a);
Node *node_log(Arena *arena, GraphContext *ctx, Node *a);
Node *node_pow(Arena *arena, GraphContext *ctx, Node *a, float exponent);

// Reductions
Node *node_sum(Arena *arena, GraphContext *ctx, Node *a);
Node *node_mean(Arena *arena, GraphContext *ctx, Node *a);

// Activations
Node *node_relu(Arena *arena, GraphContext *ctx, Node *a);
Node *node_sigmoid(Arena *arena, GraphContext *ctx, Node *a);
Node *node_softmax(Arena *arena, GraphContext *ctx, Node *a);
Node *node_tanh(Arena *arena, GraphContext *ctx, Node *a);
Node *node_leaky_relu(Arena *arena, GraphContext *ctx, Node *a,
                      float negative_slope = 0.01f);
Node *node_gelu(Arena *arena, GraphContext *ctx, Node *a);
Node *node_silu(Arena *arena, GraphContext *ctx, Node *a);

// Loss functions
Node *node_cross_entropy(Arena *arena, GraphContext *ctx, Node *p, Node *q);
Node *node_mse(Arena *arena, GraphContext *ctx, Node *a, Node *b);
Node *node_l1_loss(Arena *arena, GraphContext *ctx, Node *pred, Node *target);
Node *node_bce(Arena *arena, GraphContext *ctx, Node *pred, Node *target);

} // namespace gradientcore
