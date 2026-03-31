#include "include/gradientcore/base/arena.hpp"
#include "include/gradientcore/base/base.hpp"
#include "include/gradientcore/base/prng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace gradientcore;

struct matrix {
  uint32_t rows, cols;
  float *data;
};

matrix *mat_create(Arena *arena, int32_t rows, int32_t cols);
matrix *mat_load(Arena *arena, uint32_t rows, int32_t cols,
                 const char *filename);
void mat_clear(matrix *mat);
bool mat_copy(matrix *dst, matrix *src);
void mat_fill(matrix *mat, float x);
void mat_scale(matrix *mat, float scale);
float mat_sum(matrix *mat);
bool mat_add(matrix *out, const matrix *a, const matrix *b);
bool mat_sub(matrix *out, const matrix *a, const matrix *b);
bool mat_mul(matrix *out, const matrix *a, const matrix *b, bool zero_out,
             bool transpose_a, bool transpose_b);
bool mat_relu(matrix *out, const matrix *in);
bool mat_softmax(matrix *out, const matrix *in);
bool mat_cross_entropy(matrix *out, const matrix *p, const matrix *q);
bool mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad);
bool mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                          const matrix *grad);
bool mat_cross_entorpy_add_grad(matrix *p_grad, matrix *q_grad, const matrix *p,
                                const matrix *q, const matrix *grad);

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

struct model_training_desc {
  matrix *train_images;
  matrix *train_labels;
  matrix *test_images;
  matrix *test_labels;

  uint32_t epochs;
  uint32_t batch_size;
  float learning_rate;
};

model_var *mv_create(Arena *arena, model_context *model, uint32_t rows,
                     uint32_t cols, uint32_t flags, model_var_op op);

model_var *mv_relu(Arena *arena, model_context *model, model_var *input,
                   uint32_t flags);

model_var *mv_softmax(Arena *arena, model_context *model, model_var *input,
                      uint32_t flags);

model_var *mv_add(Arena *arena, model_context *model, model_var *a,
                  model_var *b, uint32_t flags);

model_var *mv_sub(Arena *arena, model_context *model, model_var *a,
                  model_var *b, uint32_t flags);

model_var *mv_matmul(Arena *arena, model_context *model, model_var *a,
                     model_var *b, uint32_t flags);

model_var *mv_cross_entopy(Arena *arena, model_context *model, model_var *p,
                           model_var *q, uint32_t flags);

model_program model_prog_create(Arena *arena, model_context *model,
                                model_var *out_var);
void model_program_compute(model_program *prog);
void model_program_compute_grads(model_program *prog);

model_context *model_create(Arena *arena);
void model_compile(Arena *arena, model_context *model);
void model_feedforward(model_context *model);
void model_train(model_context *model,
                 const model_training_desc *training_desc);

void draw_mnist_digit(float *data);

int main() {
  Arena *perm_arena = Arena::create(MiB(1024), MiB(1), false);

  matrix *train_images =
      mat_load(perm_arena, 60000, 784, "../data/train_images.mat");
  matrix *test_images =
      mat_load(perm_arena, 10000, 784, "../data/test_images.mat");
  matrix *train_labels = mat_create(perm_arena, 60000, 10);
  matrix *test_labels = mat_create(perm_arena, 10000, 10);

  {
    matrix *train_labels_file =
        mat_load(perm_arena, 60000, 1, "../data/train_labels.mat");
    matrix *test_labels_file =
        mat_load(perm_arena, 10000, 1, "../data/test_labels.mat");

    for (uint32_t i = 0; i < 60000; i++) {
      uint32_t num = train_labels_file->data[i];
      train_labels->data[i * 10 + num] = 1.0f;
    }

    for (uint32_t i = 0; i < 10000; i++) {
      uint32_t num = test_labels_file->data[i];
      test_labels->data[i * 10 + num] = 1.0f;
    }
  }

  perm_arena->destroy();
  return 0;
}

void draw_mnist_digit(float *data) {
  for (uint32_t y = 0; y < 28; y++) {
    for (uint32_t x = 0; x < 28; x++) {
      float num = data[x + y * 28];
      uint32_t col = 232 + (uint32_t)(num * 24);
      printf("\x1b[48;5;%dm  ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m");
}

matrix *mat_create(Arena *arena, int32_t rows, int32_t cols) {
  if (arena == nullptr || rows <= 0 || cols <= 0) {
    return nullptr;
  }

  matrix *mat = arena->push<matrix>();
  mat->rows = rows;
  mat->cols = cols;
  mat->data = arena->push_array<float>((uint64_t)rows * cols);

  return mat;
}

matrix *mat_load(Arena *arena, uint32_t rows, int32_t cols,
                 const char *filename) {
  matrix *mat = mat_create(arena, rows, cols);

  FILE *f = fopen(filename, "rb");

  if (f == nullptr) {
    std::fprintf(stderr, "Fatal Error: Could not find or open file: %s\n",
                 filename);
    std::fprintf(stderr, "Make sure the 'data' folder is in the same directory "
                         "you are running ./test_app from!\n");
    std::exit(1); // Stop the program safely
  }

  fseek(f, 0, SEEK_END);
  int64_t size = ftell(f);
  fseek(f, 0, SEEK_SET);

  size = std::min<int64_t>(size, sizeof(float) * rows * cols);

  fread(mat->data, 1, size, f);

  fclose(f);

  return mat;
}

bool mat_copy(matrix *dst, matrix *src) {
  if (dst->rows != src->rows || dst->cols != src->cols) {
    return false;
  }

  memcpy(dst->data, src->data, sizeof(float) * (uint64_t)dst->rows * dst->cols);

  return true;
}

void mat_clear(matrix *mat) {
  memset(mat->data, 0, sizeof(float) * (uint64_t)mat->rows * mat->cols);
}

void mat_fill(matrix *mat, float x) {
  uint64_t size = (uint64_t)mat->rows * mat->cols;
  for (uint64_t i = 0; i < size; i++) {
    mat->data[i] = x;
  }
}

void mat_scale(matrix *mat, float scale) {
  uint64_t size = (uint64_t)mat->rows * mat->cols;
  for (uint64_t i = 0; i < size; i++) {
    mat->data[i] *= scale;
  }
}

float mat_sum(matrix *mat) {
  uint64_t size = (uint64_t)mat->rows * mat->cols;
  float sum = 0.0;
  for (uint64_t i = 0; i < size; i++) {
    sum += mat->data[i];
  }

  return sum;
}

bool mat_add(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols)
    return false;
  if (out->rows != a->rows || out->cols != a->cols)
    return false;

  uint64_t size = (uint64_t)out->rows * out->cols;
  for (uint64_t i = 0; i < size; i++) {
    out->data[i] = a->data[i] + b->data[i];
  }

  return true;
}

bool mat_sub(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols)
    return false;
  if (out->rows != a->rows || out->cols != a->cols)
    return false;

  uint64_t size = (uint64_t)out->rows * out->cols;
  for (uint64_t i = 0; i < size; i++) {
    out->data[i] = a->data[i] - b->data[i];
  }

  return true;
}

void mat_mul_nn(matrix *out, const matrix *a, const matrix *b) {
  for (uint64_t i = 0; i < out->rows; i++) {
    for (int64_t k = 0; k < a->cols; k++) {
      for (uint64_t j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}
void mat_mul_nt(matrix *out, const matrix *a, const matrix *b) {
  for (uint64_t i = 0; i < out->rows; i++) {
    for (uint64_t j = 0; j < out->cols; j++) {
      for (int64_t k = 0; k < a->cols; k++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}
void mat_mul_tn(matrix *out, const matrix *a, const matrix *b) {
  for (int64_t k = 0; k < a->rows; k++) {
    for (uint64_t i = 0; i < out->rows; i++) {
      for (uint64_t j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}
void mat_mul_tt(matrix *out, const matrix *a, const matrix *b) {
  for (uint64_t i = 0; i < out->rows; i++) {
    for (uint64_t j = 0; j < out->cols; j++) {
      for (int64_t k = 0; k < a->rows; k++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

bool mat_mul(matrix *out, const matrix *a, const matrix *b, bool zero_out,
             bool transpose_a, bool transpose_b) {
  uint32_t a_rows = transpose_a ? a->cols : a->rows;
  uint32_t a_cols = transpose_a ? a->rows : a->cols;
  uint32_t b_cols = transpose_b ? b->rows : b->cols;
  uint32_t b_rows = transpose_b ? b->cols : b->rows;

  if (a_cols != b_rows)
    return false;
  if (out->rows != a_rows || out->cols != b_cols)
    return false;

  if (zero_out) {
    mat_clear(out);
  }

  int32_t transpose = (transpose_a << 1) | transpose_b;

  switch (transpose) {
  case 0b00: {
    mat_mul_nn(out, a, b);
  } break;
  case 0b01: {
    mat_mul_nt(out, a, b);
  } break;
  case 0b10: {
    mat_mul_tn(out, a, b);
  } break;
  case 0b11: {
    mat_mul_tt(out, a, b);

  } break;
  }
  return true;
}

bool mat_relu(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols)
    return false;

  uint64_t size = (uint64_t)out->rows * out->cols;
  for (uint64_t i = 0; i < size; i++) {
    out->data[i] = std::max(0.0f, in->data[i]);
  }

  return true;
}

bool mat_softmax(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols)
    return false;

  uint64_t size = (uint64_t)out->rows * out->cols;
  if (size == 0) {
    return false;
  }

  float max_logit = in->data[0];
  for (uint64_t i = 1; i < size; i++) {
    max_logit = std::max(max_logit, in->data[i]);
  }

  float sum = 0.0f;
  for (uint64_t i = 0; i < size; i++) {
    out->data[i] = std::exp(in->data[i] - max_logit);
    sum += out->data[i];
  }

  if (sum <= 0.0f || !std::isfinite(sum)) {
    return false;
  }

  mat_scale(out, 1.0f / sum);

  return true;
}

bool mat_cross_entropy(matrix *out, const matrix *p, const matrix *q) {
  if (p->rows != q->rows || p->cols != q->cols)
    return false;
  if (out->rows != p->rows || out->cols != p->cols)
    return false;

  uint64_t size = (uint64_t)out->rows * out->cols;
  for (uint64_t i = 0; i < size; i++) {
    if (p->data[i] == 0.0f) {
      out->data[i] = 0.0f;
      continue;
    }

    float qi = q->data[i];
    if (!(qi > 0.0f)) {
      qi = 1e-12f;
    }
    out->data[i] = -p->data[i] * logf(qi);
  }

  return true;
}

bool mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }
  if (out->rows != grad->rows || out->cols != grad->cols) {
    return false;
  }

  uint64_t size = (uint64_t)out->rows * out->cols;
  for (uint64_t i = 0; i < size; i++) {
    out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
  }

  return true;
}

bool mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                          const matrix *grad) {
  if (softmax_out->rows != 1 && softmax_out->cols != 1) {
    return false;
  }

  ArenaTemp scratch = scratch_get(NULL, 0);

  uint32_t size = std::max(softmax_out->rows, softmax_out->cols);
  matrix *jacobian = mat_create(scratch.arena, size, size);

  for (uint32_t i = 0; i < size; i++) {
    for (uint32_t j = 0; j < size; j++) {
      jacobian->data[j + i * size] =
          softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
    }
  }

  mat_mul(out, jacobian, grad, 0, 0, 0);

  return true;
}

bool mat_cross_entorpy_add_grad(matrix *p_grad, matrix *q_grad, const matrix *p,
                                const matrix *q, const matrix *grad) {
  if (p->rows != q->rows || p->cols != q->cols)
    return false;

  uint64_t size = (uint64_t)p->rows * p->cols;

  if (p_grad != NULL) {
    if (p_grad->rows != p->rows || p_grad->cols != p->cols) {
      return false;
    }

    for (uint64_t i = 0; i < size; i++) {
      p_grad->data[i] += (-1) * logf(q->data[i]) * grad->data[i];
    }
  }

  if (q_grad != NULL) {
    if (q_grad->rows != q->rows || q_grad->cols != q->cols) {
      return false;
    }

    for (uint64_t i = 0; i < size; i++) {
      q_grad->data[i] += (-1) * (p->data[i] / q->data[i]) * grad->data[i];
    }
  }

  return true;
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

model_program model_prog_create(Arena *arena, model_context *model,
                                model_var *out_var) {
  ArenaTemp scratch = scratch_get(&arena, 1);

  bool *visited = scratch.arena->push_array<bool>(model->num_vars);
  uint32_t stack_size = 0;
  uint32_t out_size;
  model_var **stack = scratch.arena->push_array<model_var *>(model->num_vars);
  model_var **out = scratch.arena->push_array<model_var *>(model->num_vars);

  stack[stack_size++] = out_var;

  while (stack_size > 0) {
    model_var *cur = stack[--stack_size];

    if (cur->index >= model->num_vars) {
      continue;
    }

    if (visited[cur->index]) {
      if (out_size < model->num_vars) {
        out[out_size++] = cur;
      }
      continue;
    }

    visited[cur->index] = true;

    if (stack_size < model->num_vars) {
      stack[stack_size++] = cur;
    }

    uint32_t num_inputs = MY_NUM_INPUTS(cur->op);
    for (uint32_t i = 0; i < num_inputs; i++) {
      model_var *input = cur->inputs[i];

      if (input->index >= model->num_vars || visited[input->index]) {
        continue;
      }
      for (uint32_t j = 0; j < stack_size; j++) {
        if (stack[j] == input) {
          for (uint32_t k = j; k < stack_size - 1; k++) {
            stack[k] = stack[k + 1];
          }
          stack_size--;
        }
      }

      if (stack_size < model->num_vars) {
        stack[stack_size++] = input;
      }
    }
  }

  model_program prog = {};
  prog.size = out_size;
  prog.vars = arena->push_array<model_var *>(out_size);

  memcpy(prog.vars, out, sizeof(model_var *) * out_size);

  return prog;
}

void model_prog_compute(model_program *prog) {
  for (uint32_t i = 0; i < prog->size; i++) {
    model_var *cur = prog->vars[i];

    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    switch (cur->op) {
    case MV_OP_NULL:
    case MV_OP_CREATE:
      break;

    case _MV_OP_UNARY_START:
      break;

    case MV_OP_RELU: {
      mat_relu(cur->val, a->val);
    } break;

    case MV_OP_SOFTMAX: {
      mat_softmax(cur->val, a->val);
    } break;

    case _MV_OP_BINARY_START:
      break;

    case MV_OP_ADD: {
      mat_add(cur->val, a->val, b->val);
    } break;

    case MV_OP_SUB: {
      mat_sub(cur->val, a->val, b->val);
    } break;

    case MV_OP_MATMUL: {
      mat_mul(cur->val, a->val, b->val, true, false, false);
    } break;

    case MV_OP_CROSS_ENTROPY: {
      mat_cross_entropy(cur->val, a->val, b->val);
    } break;
    }
  }
}

void model_program_compute_grads(model_program *prog) {
  for (uint32_t i = 0; i < prog->size; i++) {
    model_var *cur = prog->vars[i];

    if ((cur->flags & MY_FLAG_REQUIRES_GRAD) != MY_FLAG_REQUIRES_GRAD) {
      continue;
    }

    if ((cur->flags & MY_FLAG_REQUIRES_GRAD)) {
      continue;
    }

    mat_clear(cur->grad);
  }

  mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

  for (int64_t i = (int64_t)prog->size - 1; i >= 0; i--) {
    model_var *cur = prog->vars[i];

    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    uint32_t num_inputs = MY_NUM_INPUTS(cur->op);

    if (num_inputs == 1 &&
        (a->flags & MY_FLAG_REQUIRES_GRAD) != MY_FLAG_REQUIRES_GRAD) {
      continue;
    }

    if (num_inputs == 2 &&
        (a->flags & MY_FLAG_REQUIRES_GRAD) != MY_FLAG_REQUIRES_GRAD &&
        (b->flags & MY_FLAG_REQUIRES_GRAD) != MY_FLAG_REQUIRES_GRAD) {
      continue;
    }

    switch (cur->op) {
    case MV_OP_NULL:
    case MV_OP_CREATE:
      break;

    case _MV_OP_UNARY_START:
      break;

    case MV_OP_RELU: {
      mat_relu_add_grad(a->grad, a->val, cur->grad);
    } break;

    case MV_OP_SOFTMAX: {
      mat_softmax_add_grad(a->grad, cur->val, cur->grad);
    } break;

    case _MV_OP_BINARY_START:
      break;

    case MV_OP_ADD: {
      if (a->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_add(a->grad, a->grad, cur->grad);
      }
      if (b->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_add(b->grad, b->grad, cur->grad);
      }
    } break;

    case MV_OP_SUB: {
      if (a->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_add(a->grad, a->grad, cur->grad);
      }
      if (b->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_sub(b->grad, b->grad, cur->grad);
      }

    } break;

    case MV_OP_MATMUL: {
      if (a->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
      }
      if (b->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
      }
    } break;

    case MV_OP_CROSS_ENTROPY: {
      model_var *p = a;
      model_var *q = b;

      mat_cross_entorpy_add_grad(p->grad, q->grad, p->val, q->val, cur->grad);
    } break;
    }
  }
}
