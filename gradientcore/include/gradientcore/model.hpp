#pragma once

#include "matrix.hpp"
#include <cstdint>

namespace gradientcore {

enum model_var_flags {
  MY_FLAG_NONE = 0,
  MY_FLAG_REQUIRES_GRAD = (1 << 0),
  MY_FLAG_PARAMETER = (1 << 1),
  MY_FLAG_INPUT = (1 << 2),
  MY_FLAG_OUTPUT = (1 << 3),
  MY_FLAG_DESIRED_OUTPUT = (1 << 4),
  MY_FLAG_COST = (1 << 5),
};

enum model_var_op {
  MV_OP_NULL,
  MV_OP_CREATE,

  _MV_OP_UNARY_START,

  MV_OP_RELU,
  MV_OP_SOFTMAX,

  _MV_OP_BINARY_START,

  MV_OP_ADD,
  MV_OP_SUB,
  MV_OP_MATMUL,
  MV_OP_CROSS_ENTROPY,
};

#define MODEL_VAR_MAX_INPUTS 2
#define MY_NUM_INPUTS(op)                                                      \
  ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

struct model_var {
  uint32_t index;
  uint32_t flags;

  matrix *val;
  matrix *grad;

  model_var_op op;
  struct model_var *inputs[MODEL_VAR_MAX_INPUTS];
};

struct model_program {
  model_var **vars;
  uint32_t size;
};

struct model_context {
  uint32_t num_vars;

  model_var *input;
  model_var *output;
  model_var *desired_output;
  model_var *cost;

  model_program forward_prog;
  model_program cost_prog;
};

// Model creation
model_context *model_create(class Arena *arena);

// Model variable creation
model_var *mv_create(class Arena *arena, model_context *model, uint32_t rows,
                     uint32_t cols, uint32_t flags);

bool model_save_weights(model_context *model, const char *filename);
bool model_load_weights(model_context *model, const char *filename);

} // namespace gradientcore
