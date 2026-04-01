#include "../../include/gradientcore/serialize/serialize.hpp"
#include <cstdio>
#include <cstring>

namespace gradientcore {
namespace serialize {

static constexpr uint32_t MAGIC = 0x47435746; // "GCWF" — GradientCore Weight File

bool save_weights(const char *path, Node **params, uint32_t num_params) {
  FILE *f = std::fopen(path, "wb");
  if (f == nullptr)
    return false;

  // Header
  std::fwrite(&MAGIC, sizeof(uint32_t), 1, f);
  std::fwrite(&num_params, sizeof(uint32_t), 1, f);

  for (uint32_t i = 0; i < num_params; i++) {
    Tensor *t = params[i]->val;
    if (t == nullptr) {
      std::fclose(f);
      return false;
    }

    // Write tensor metadata
    std::fwrite(&t->ndims, sizeof(uint32_t), 1, f);
    std::fwrite(t->shape, sizeof(uint32_t), t->ndims, f);

    // Write tensor data (handle non-contiguous tensors)
    if (tensor_is_contiguous(t)) {
      std::fwrite(t->storage->data + t->offset, sizeof(float), t->size, f);
    } else {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t j = 0; j < t->size; j++) {
        uint64_t idx = tensor_get_flat_index(t, indices);
        std::fwrite(&t->storage->data[idx], sizeof(float), 1, f);
        for (int32_t d = t->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < t->shape[d])
            break;
          indices[d] = 0;
        }
      }
    }
  }

  std::fclose(f);
  return true;
}

bool load_weights(const char *path, Node **params, uint32_t num_params) {
  FILE *f = std::fopen(path, "rb");
  if (f == nullptr)
    return false;

  // Validate header
  uint32_t magic = 0;
  std::fread(&magic, sizeof(uint32_t), 1, f);
  if (magic != MAGIC) {
    std::fprintf(stderr, "Error: invalid weight file (bad magic)\n");
    std::fclose(f);
    return false;
  }

  uint32_t saved_num_params = 0;
  std::fread(&saved_num_params, sizeof(uint32_t), 1, f);
  if (saved_num_params != num_params) {
    std::fprintf(stderr,
                 "Error: param count mismatch (file=%u, expected=%u)\n",
                 saved_num_params, num_params);
    std::fclose(f);
    return false;
  }

  for (uint32_t i = 0; i < num_params; i++) {
    Tensor *t = params[i]->val;
    if (t == nullptr) {
      std::fclose(f);
      return false;
    }

    // Read and validate shape
    uint32_t ndims = 0;
    std::fread(&ndims, sizeof(uint32_t), 1, f);
    if (ndims != t->ndims) {
      std::fprintf(stderr, "Error: ndims mismatch for param %u\n", i);
      std::fclose(f);
      return false;
    }

    uint32_t shape[MAX_TENSOR_DIMS] = {0};
    std::fread(shape, sizeof(uint32_t), ndims, f);
    for (uint32_t d = 0; d < ndims; d++) {
      if (shape[d] != t->shape[d]) {
        std::fprintf(stderr, "Error: shape mismatch for param %u dim %u\n", i,
                     d);
        std::fclose(f);
        return false;
      }
    }

    // Read tensor data
    if (tensor_is_contiguous(t)) {
      std::fread(t->storage->data + t->offset, sizeof(float), t->size, f);
    } else {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t j = 0; j < t->size; j++) {
        uint64_t idx = tensor_get_flat_index(t, indices);
        std::fread(&t->storage->data[idx], sizeof(float), 1, f);
        for (int32_t d = t->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < t->shape[d])
            break;
          indices[d] = 0;
        }
      }
    }
  }

  std::fclose(f);
  return true;
}

} // namespace serialize
} // namespace gradientcore
