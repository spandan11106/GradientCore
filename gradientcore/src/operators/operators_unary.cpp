#include "../../include/gradientcore/operators.hpp"
#include "../../include/gradientcore/base/arena.hpp"

namespace gradientcore {

model_var *_my_unary_impl(Arena *arena, model_context *model, model_var *input,
                          uint32_t rows, uint32_t cols, uint32_t flags,
                          model_var_op op) {
  if (input->flags & MY_FLAG_REQUIRES_GRAD) {
    flags |= MY_FLAG_REQUIRES_GRAD;
  }

  model_var *out = mv_create(arena, model, rows, cols, flags);
  out->op = op;
  out->inputs[0] = input;

  return out;
}

model_var *mv_relu(Arena *arena, model_context *model, model_var *input,
                   uint32_t flags) {
  return _my_unary_impl(arena, model, input, input->val->rows, input->val->cols,
                        flags, MV_OP_RELU);
}

model_var *mv_softmax(Arena *arena, model_context *model, model_var *input,
                      uint32_t flags) {
  return _my_unary_impl(arena, model, input, input->val->rows, input->val->cols,
                        flags, MV_OP_SOFTMAX);
}

} // namespace gradientcore
