#include "include/gradientcore/model.hpp"
#include "include/gradientcore/matrix.hpp"
#include "include/gradientcore/base/arena.hpp"

namespace gradientcore {

model_context *model_create(Arena *arena) {
  model_context *model = arena->push<model_context>();
  return model;
}

model_var *mv_create(Arena *arena, model_context *model, uint32_t rows,
                     uint32_t cols, uint32_t flags) {
  model_var *out = arena->push<model_var>();

  out->index = model->num_vars++;
  out->flags = flags;
  out->op = MV_OP_CREATE;
  out->val = mat_create(arena, rows, cols);

  if (flags & MY_FLAG_REQUIRES_GRAD) {
    out->grad = mat_create(arena, rows, cols);
  }

  if (flags & MY_FLAG_INPUT) {
    model->input = out;
  }
  if (flags & MY_FLAG_OUTPUT) {
    model->output = out;
  }
  if (flags & MY_FLAG_DESIRED_OUTPUT) {
    model->desired_output = out;
  }
  if (flags & MY_FLAG_COST) {
    model->cost = out;
  }

  return out;
}

} // namespace gradientcore
