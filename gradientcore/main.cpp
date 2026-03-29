#include "include/gradientcore/base/arena.hpp"
#include "include/gradientcore/base/base.hpp"
#include "include/gradientcore/base/prng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

using namespace gradientcore;

struct matrix {
  uint32_t rows, cols;
  float *data;
};

matrix *mat_create(Arena *arena, int32_t rows, int32_t cols);
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
bool mat_relu_add_grad(matrix *out, const matrix *in);
bool mat_softmax_add_grad(matrix *out, const matrix *softmax_out);
bool mat_cross_entorpy_add_grad(matrix *out, const matrix *p, const matrix *q);

int main() {
  Arena *perm_arena = Arena::create(MiB(1024), MiB(1), false);
  if (perm_arena != nullptr) {
    perm_arena->destroy();
  }

  return 0;
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
      for (int64_t k = 0; k < a->cols; k++) {
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
