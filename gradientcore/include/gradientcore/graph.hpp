#pragma once

#include "model.hpp"

namespace gradientcore {

// Graph compilation and execution
model_program model_prog_create(class Arena *arena, model_context *model,
                                model_var *out_var);

void model_prog_compute(model_program *prog);

void model_program_compute_grads(model_program *prog);

void model_compile(class Arena *arena, model_context *model);

void model_feedforward(model_context *model);

} // namespace gradientcore
