#pragma once

#include "../autograd/autograd.hpp"
#include "../core/tensor.hpp"
#include <cstdint>

namespace gradientcore {
namespace serialize {

// Saves all parameter value tensors to a binary file.
// Format: [magic][num_params][for each: ndims, shape[], data[]]
bool save_weights(const char *path, Node **params, uint32_t num_params);

// Loads weights from a binary file into existing parameter nodes.
// The parameter shapes must match the saved shapes exactly.
bool load_weights(const char *path, Node **params, uint32_t num_params);

} // namespace serialize
} // namespace gradientcore
