#pragma once

#include "../autograd/autograd.hpp"
#include "../ops/ops.hpp"

namespace gradientcore {
namespace nn {

void init_uniform(Tensor *t, float bound);

void init_kaiming_uniform(Tensor *t, uint32_t fan_in);

void init_xavier_uniform(Tensor *t, uint32_t fan_in, uint32_t fan_out);

struct Linear {
  Node *weight;
  Node *bias;
};

Linear linear_create(Arena *arena, GraphContext *ctx, uint32_t in_features,
                     uint32_t out_features, bool use_bias = true);

Node *linear_forward(Arena *arena, GraphContext *ctx, Linear layer,
                     Node *input);

} // namespace nn
} // namespace gradientcore
