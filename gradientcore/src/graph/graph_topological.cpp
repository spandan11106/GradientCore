#include "../../include/gradientcore/graph.hpp"
#include "../../include/gradientcore/base/arena.hpp"

#include <cstring>

namespace gradientcore {

model_program model_prog_create(Arena *arena, model_context *model,
                                model_var *out_var) {
  ArenaTemp scratch = scratch_get(&arena, 1);

  bool *visited = scratch.arena->push_array<bool>(model->num_vars);
  memset(visited, 0, sizeof(bool) * model->num_vars);
  uint32_t stack_size = 0;
  uint32_t out_size = 0;
  model_var **stack = scratch.arena->push_array<model_var *>(model->num_vars);
  model_var **out = scratch.arena->push_array<model_var *>(model->num_vars);

  stack[stack_size++] = out_var;

  while (stack_size > 0) {
    model_var *cur = stack[--stack_size];

    if (cur->index >= model->num_vars) {
      continue;
    }

    if (visited[cur->index]) {
      if (out_size < model->num_vars) {
        out[out_size++] = cur;
      }
      continue;
    }

    visited[cur->index] = true;

    if (stack_size < model->num_vars) {
      stack[stack_size++] = cur;
    }

    uint32_t num_inputs = MY_NUM_INPUTS(cur->op);
    for (uint32_t i = 0; i < num_inputs; i++) {
      model_var *input = cur->inputs[i];

      if (input->index >= model->num_vars || visited[input->index]) {
        continue;
      }
      for (uint32_t j = 0; j < stack_size; j++) {
        if (stack[j] == input) {
          for (uint32_t k = j; k < stack_size - 1; k++) {
            stack[k] = stack[k + 1];
          }
          stack_size--;
        }
      }

      if (stack_size < model->num_vars) {
        stack[stack_size++] = input;
      }
    }
  }

  model_program prog = {};
  prog.size = out_size;
  prog.vars = arena->push_array<model_var *>(out_size);

  memcpy(prog.vars, out, sizeof(model_var *) * out_size);

  return prog;
}

} // namespace gradientcore
