#include "../../include/gradientcore/matrix.hpp"

namespace gradientcore {

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

uint32_t mat_argmax(matrix *mat) {
  uint64_t size = (uint64_t)mat->rows * mat->cols;
  uint64_t max_i = 0;
  for (uint64_t i = 0; i < size; i++) {
    if (mat->data[i] > mat->data[max_i]) {
      max_i = i;
    }
  }

  return max_i;
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

} // namespace gradientcore
