#include "../../include/gradientcore/matrix.hpp"

namespace gradientcore {

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

} // namespace gradientcore
