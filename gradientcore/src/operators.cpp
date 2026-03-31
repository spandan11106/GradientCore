#include "include/gradientcore/operators.hpp"
#include "include/gradientcore/base/arena.hpp"

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

model_var *_my_binary_impl(Arena *arena, model_context *model, model_var *a,
                           model_var *b, uint32_t rows, uint32_t cols,
                           uint32_t flags, model_var_op op) {
  if ((a->flags & MY_FLAG_REQUIRES_GRAD) ||
      (b->flags & MY_FLAG_REQUIRES_GRAD)) {
    flags |= MY_FLAG_REQUIRES_GRAD;
  }

  model_var *out = mv_create(arena, model, rows, cols, flags);
  out->op = op;
  out->inputs[0] = a;
  out->inputs[1] = b;

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

model_var *mv_add(Arena *arena, model_context *model, model_var *a,
                  model_var *b, uint32_t flags) {
  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    return NULL;
  }

  return _my_binary_impl(arena, model, a, b, a->val->rows, a->val->cols, flags,
                         MV_OP_ADD);
}

model_var *mv_sub(Arena *arena, model_context *model, model_var *a,
                  model_var *b, uint32_t flags) {
  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    return NULL;
  }

  return _my_binary_impl(arena, model, a, b, a->val->rows, a->val->cols, flags,
                         MV_OP_SUB);
}

model_var *mv_matmul(Arena *arena, model_context *model, model_var *a,
                     model_var *b, uint32_t flags) {
  if (a->val->cols != b->val->rows) {
    return NULL;
  }

  return _my_binary_impl(arena, model, a, b, a->val->rows, b->val->cols, flags,
                         MV_OP_MATMUL);
}

model_var *mv_cross_entopy(Arena *arena, model_context *model, model_var *p,
                           model_var *q, uint32_t flags) {
  if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
    return NULL;
  }

  return _my_binary_impl(arena, model, p, q, p->val->rows, p->val->cols, flags,
                         MV_OP_CROSS_ENTROPY);
}

} // namespace gradientcore
