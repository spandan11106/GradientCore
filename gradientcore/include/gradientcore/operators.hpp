#pragma once

#include "model.hpp"

namespace gradientcore {

// Unary operations
model_var *mv_relu(class Arena *arena, model_context *model, model_var *input,
                   uint32_t flags);

model_var *mv_softmax(class Arena *arena, model_context *model, model_var *input,
                      uint32_t flags);

// Binary operations
model_var *mv_add(class Arena *arena, model_context *model, model_var *a,
                  model_var *b, uint32_t flags);

model_var *mv_sub(class Arena *arena, model_context *model, model_var *a,
                  model_var *b, uint32_t flags);

model_var *mv_matmul(class Arena *arena, model_context *model, model_var *a,
                     model_var *b, uint32_t flags);

model_var *mv_cross_entopy(class Arena *arena, model_context *model, model_var *p,
                           model_var *q, uint32_t flags);

} // namespace gradientcore
