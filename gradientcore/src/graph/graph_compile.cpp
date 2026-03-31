#include "../../include/gradientcore/graph.hpp"

#include <cstddef>

namespace gradientcore {

void model_compile(Arena *arena, model_context *model) {
  if (model->output != NULL) {
    model->forward_prog = model_prog_create(arena, model, model->output);
  }
  if (model->cost != NULL) {
    model->cost_prog = model_prog_create(arena, model, model->cost);
  }
}

void model_feedforward(model_context *model) {
  model_prog_compute(&model->forward_prog);
}

} // namespace gradientcore
