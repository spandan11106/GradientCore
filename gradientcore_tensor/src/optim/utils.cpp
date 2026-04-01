#include "../../include/gradientcore/optim/optim.hpp"

namespace gradientcore {
namespace optim {

void zero_grad(Node **params, uint32_t num_params) {
  for (uint32_t i = 0; i < num_params; i++) {
    if (params[i] != nullptr && params[i]->grad != nullptr) {
      tensor_clear(params[i]->grad);
    }
  }
}

} // namespace optim
} // namespace gradientcore
