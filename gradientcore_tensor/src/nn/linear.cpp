#include "../../include/gradientcore/nn/nn.hpp"

namespace gradientcore {
namespace nn {

Linear linear_create(Arena *arena, GraphContext *ctx, uint32_t in_features,
                     uint32_t out_features, bool use_bias) {
  Linear layer = {};

  uint32_t w_shape[2] = {in_features, out_features};
  layer.weight = node_create(arena, ctx, 2, w_shape,
                             NODE_FLAG_PARAMETER | NODE_FLAG_REQUIRES_GRAD);
  if (layer.weight == nullptr)
    return layer;

  init_kaiming_uniform(layer.weight->val, in_features);

  if (use_bias) {
    uint32_t b_shape[1] = {out_features};
    layer.bias = node_create(arena, ctx, 1, b_shape,
                             NODE_FLAG_PARAMETER | NODE_FLAG_REQUIRES_GRAD);
    if (layer.bias == nullptr)
      return layer;

    tensor_clear(layer.bias->val);
  } else {
    layer.bias = nullptr;
  }

  return layer;
}

Node *linear_forward(Arena *arena, GraphContext *ctx, Linear layer,
                     Node *input) {
  Node *out = node_matmul(arena, ctx, input, layer.weight);
  if (out == nullptr)
    return nullptr;

  if (layer.bias != nullptr) {
    out = node_add(arena, ctx, out, layer.bias);
  }

  return out;
}

} // namespace nn
} // namespace gradientcore
