#include "../../include/gradientcore/matrix.hpp"
#include "../../include/gradientcore/base/arena.hpp"
#include "../../include/gradientcore/base/prng.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace gradientcore {

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
    std::exit(1);
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

void mat_fill_rand(matrix *mat, float lower, float upper) {
  uint64_t size = (uint64_t)mat->rows * mat->cols;
  for (uint64_t i = 0; i < size; i++) {
    mat->data[i] = prng::randf() * (upper - lower) + lower;
  }
}

} // namespace gradientcore
