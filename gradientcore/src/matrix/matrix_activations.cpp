#include "../../include/gradientcore/matrix.hpp"
#include "../../include/gradientcore/base/arena.hpp"

#include <algorithm>
#include <cmath>

namespace gradientcore {

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
      float qi = q->data[i];
      if (!(qi > 0.0f)) {
        qi = 1e-12f;
      }
      q_grad->data[i] += (-1) * (p->data[i] / qi) * grad->data[i];
    }
  }

  return true;
}

} // namespace gradientcore
