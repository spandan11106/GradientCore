#pragma once

#include <cstdint>

namespace gradientcore {

struct matrix {
  uint32_t rows, cols;
  float *data;
};

// Matrix creation and loading
matrix *mat_create(class Arena *arena, int32_t rows, int32_t cols);
matrix *mat_load(class Arena *arena, uint32_t rows, int32_t cols,
                 const char *filename);

// Matrix operations
void mat_clear(matrix *mat);
bool mat_copy(matrix *dst, matrix *src);
void mat_fill(matrix *mat, float x);
void mat_fill_rand(matrix *mat, float lower, float upper);
void mat_scale(matrix *mat, float scale);
float mat_sum(matrix *mat);
uint32_t mat_argmax(matrix *mat);

// Arithmetic operations
bool mat_add(matrix *out, const matrix *a, const matrix *b);
bool mat_sub(matrix *out, const matrix *a, const matrix *b);
bool mat_mul(matrix *out, const matrix *a, const matrix *b, bool zero_out,
             bool transpose_a, bool transpose_b);

// Activation functions
bool mat_relu(matrix *out, const matrix *in);
bool mat_softmax(matrix *out, const matrix *in);

// Loss and gradients
bool mat_cross_entropy(matrix *out, const matrix *p, const matrix *q);
bool mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad);
bool mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                          const matrix *grad);
bool mat_cross_entorpy_add_grad(matrix *p_grad, matrix *q_grad, const matrix *p,
                                const matrix *q, const matrix *grad);

} // namespace gradientcore
