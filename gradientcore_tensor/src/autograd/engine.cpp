#include "../../include/gradientcore/autograd/engine.hpp"
#include "../../include/gradientcore/ops/ops.hpp"
#include <cstring>

namespace gradientcore {

GraphProgram graph_compile(Arena *arena, GraphContext *ctx, Node *out_node) {
  ArenaTemp scratch = scratch_get(&arena, 1);

  bool *visited = scratch.arena->push_array<bool>(ctx->num_nodes, false);
  Node **stack = scratch.arena->push_array<Node *>(ctx->num_nodes);
  Node **out = scratch.arena->push_array<Node *>(ctx->num_nodes);

  uint32_t stack_size = 0;
  uint32_t out_size = 0;

  stack[stack_size++] = out_node;

  while (stack_size > 0) {
    Node *cur = stack[--stack_size];

    if (cur == nullptr || cur->index >= ctx->num_nodes)
      continue;

    if (visited[cur->index]) {
      // If we've seen it and it's popping off the stack again, it belongs in
      // the sorted output
      if (out_size < ctx->num_nodes) {
        out[out_size++] = cur;
      }
      continue;
    }

    visited[cur->index] = true;

    // Push back onto stack so we process it after its children
    if (stack_size < ctx->num_nodes) {
      stack[stack_size++] = cur;
    }

    // Push children onto the stack
    uint32_t num_inputs = NUM_INPUTS(cur->op);
    for (uint32_t i = 0; i < num_inputs; i++) {
      Node *input = cur->inputs[i];

      if (input == nullptr || input->index >= ctx->num_nodes ||
          visited[input->index]) {
        continue;
      }

      // Remove from deeper in the stack if it exists, to push it to the top
      for (uint32_t j = 0; j < stack_size; j++) {
        if (stack[j] == input) {
          for (uint32_t k = j; k < stack_size - 1; k++) {
            stack[k] = stack[k + 1];
          }
          stack_size--;
          break;
        }
      }

      if (stack_size < ctx->num_nodes) {
        stack[stack_size++] = input;
      }
    }
  }

  GraphProgram prog = {};
  prog.size = out_size;
  prog.nodes = arena->push_array<Node *>(out_size);
  std::memcpy(prog.nodes, out, sizeof(Node *) * out_size);

  return prog;
}

void graph_backward(GraphProgram *prog) {
  if (prog->size == 0)
    return;

  for (uint32_t i = 0; i < prog->size; i++) {
    Node *cur = prog->nodes[i];
    if (cur->grad != nullptr && !(cur->flags & NODE_FLAG_PARAMETER)) {
      tensor_clear(cur->grad);
    }
  }

  Node *final_node = prog->nodes[prog->size - 1];
  if (final_node->grad != nullptr) {
    tensor_fill(final_node->grad, 1.0f);
  }

  for (int64_t i = (int64_t)prog->size - 1; i >= 0; i--) {
    Node *cur = prog->nodes[i];
    if (cur->grad == nullptr)
      continue;

    Node *a = cur->inputs[0];
    Node *b = cur->inputs[1];

    switch (cur->op) {

    default:
      break;
    }
  }
}

} // namespace gradientcore
