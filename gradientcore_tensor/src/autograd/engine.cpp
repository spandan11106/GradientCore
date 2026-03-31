#include "../../include/gradientcore/autograd/engine.hpp"
#include "../../include/gradientcore/ops/ops_utils.hpp"
#include <cmath>
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
      if (out_size < ctx->num_nodes) {
        out[out_size++] = cur;
      }
      continue;
    }

    visited[cur->index] = true;

    if (stack_size < ctx->num_nodes) {
      stack[stack_size++] = cur;
    }

    uint32_t num_inputs = NUM_INPUTS(cur->op);
    for (uint32_t i = 0; i < num_inputs; i++) {
      Node *input = cur->inputs[i];
      if (input == nullptr || input->index >= ctx->num_nodes ||
          visited[input->index])
        continue;

      for (uint32_t j = 0; j < stack_size; j++) {
        if (stack[j] == input) {
          for (uint32_t k = j; k < stack_size - 1; k++)
            stack[k] = stack[k + 1];
          stack_size--;
          break;
        }
      }
      if (stack_size < ctx->num_nodes)
        stack[stack_size++] = input;
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
    case OP_NULL:
    case OP_CREATE:
      break;

    // --- BASIC ARITHMETIC ---
    case OP_ADD: {
      if (a && a->grad)
        tensor_accumulate_grad(a->grad, cur->grad, 1.0f);
      if (b && b->grad)
        tensor_accumulate_grad(b->grad, cur->grad, 1.0f);
      break;
    }
    case OP_SUB: {
      if (a && a->grad)
        tensor_accumulate_grad(a->grad, cur->grad, 1.0f);
      if (b && b->grad)
        tensor_accumulate_grad(b->grad, cur->grad, -1.0f);
      break;
    }
    case OP_MUL: {
      if (a && a->grad) {
        tensor_accumulate_grad_binary_custom(
            a->grad, a->val, b->val, cur->grad,
            [](float, float b_v, float dC) { return b_v * dC; });
      }
      if (b && b->grad) {
        tensor_accumulate_grad_binary_custom(
            b->grad, a->val, b->val, cur->grad,
            [](float a_v, float, float dC) { return a_v * dC; });
      }
      break;
    }
    case OP_MATMUL: {
      if ((a && a->grad) || (b && b->grad)) {
        ArenaTemp scratch = scratch_get(nullptr, 0);

        if (a && a->grad) {
          Tensor *b_T = tensor_transpose(scratch.arena, b->val,
                                         b->val->ndims - 2, b->val->ndims - 1);
          tensor_accumulate_grad_matmul(a->grad, cur->grad, b_T);
        }
        if (b && b->grad) {
          Tensor *a_T = tensor_transpose(scratch.arena, a->val,
                                         a->val->ndims - 2, a->val->ndims - 1);
          tensor_accumulate_grad_matmul(b->grad, a_T, cur->grad);
        }
      }
      break;
    }

    // --- ACTIVATIONS ---
    case OP_RELU: {
      if (a && a->grad) {
        tensor_accumulate_grad_custom(
            a->grad, a->val, cur->grad,
            [](float x, float dC) { return (x > 0.0f ? 1.0f : 0.0f) * dC; });
      }
      break;
    }
    case OP_LEAKY_RELU: {
      if (a && a->grad) {
        // Note: using 0.01f as standard fallback here for backprop
        tensor_accumulate_grad_custom(
            a->grad, a->val, cur->grad,
            [](float x, float dC) { return (x > 0.0f ? 1.0f : 0.01f) * dC; });
      }
      break;
    }
    case OP_SIGMOID: {
      if (a && a->grad) {
        // cur->val holds sigmoid(x). d(sig)/dx = sig * (1 - sig)
        tensor_accumulate_grad_custom(
            a->grad, cur->val, cur->grad,
            [](float sig, float dC) { return sig * (1.0f - sig) * dC; });
      }
      break;
    }
    case OP_TANH: {
      if (a && a->grad) {
        // cur->val holds tanh(x). d(tanh)/dx = 1 - tanh(x)^2
        tensor_accumulate_grad_custom(
            a->grad, cur->val, cur->grad,
            [](float th, float dC) { return (1.0f - th * th) * dC; });
      }
      break;
    }
    case OP_GELU: {
      if (a && a->grad) {
        tensor_accumulate_grad_custom(
            a->grad, a->val, cur->grad, [](float x, float dC) {
              // Approximation of GELU derivative
              const float SQRT_2_OVER_PI = 0.7978845608f;
              float cdf =
                  0.5f * (1.0f + std::tanh(SQRT_2_OVER_PI *
                                           (x + 0.044715f * x * x * x)));
              float pdf =
                  std::exp(-0.5f * x * x) / 2.50662827463f; // 1/sqrt(2*pi)
              return (cdf + x * pdf) * dC;
            });
      }
      break;
    }
    case OP_SILU: {
      if (a && a->grad) {
        tensor_accumulate_grad_custom(
            a->grad, a->val, cur->grad, [](float x, float dC) {
              float sig = 1.0f / (1.0f + std::exp(-x));
              float silu = x * sig;
              return (silu + sig * (1.0f - silu)) * dC;
            });
      }
      break;
    }
    case OP_SOFTMAX: {
      if (a && a->grad) {
        tensor_accumulate_grad_softmax(a->grad, cur->val, cur->grad);
      }
      break;
    }

    // --- LOSS FUNCTIONS ---
    case OP_MSE: {
      if (a && a->grad) {
        tensor_accumulate_grad_binary_custom(
            a->grad, a->val, b->val, cur->grad,
            [](float p, float t, float dC) { return 2.0f * (p - t) * dC; });
      }
      if (b && b->grad) {
        tensor_accumulate_grad_binary_custom(
            b->grad, a->val, b->val, cur->grad,
            [](float p, float t, float dC) { return -2.0f * (p - t) * dC; });
      }
      break;
    }
    case OP_L1_LOSS: {
      if (a && a->grad) {
        tensor_accumulate_grad_binary_custom(
            a->grad, a->val, b->val, cur->grad, [](float p, float t, float dC) {
              return (p > t ? 1.0f : -1.0f) * dC;
            });
      }
      if (b && b->grad) {
        tensor_accumulate_grad_binary_custom(
            b->grad, a->val, b->val, cur->grad, [](float p, float t, float dC) {
              return (p > t ? -1.0f : 1.0f) * dC;
            });
      }
      break;
    }
    case OP_CROSS_ENTROPY: {
      if (a && a->grad) { // a = p (predictions)
        tensor_accumulate_grad_binary_custom(a->grad, a->val, b->val, cur->grad,
                                             [](float p, float t, float dC) {
                                               float pi = p > 0.0f ? p : 1e-12f;
                                               return -(t / pi) * dC;
                                             });
      }
      if (b && b->grad) { // b = t (targets)
        tensor_accumulate_grad_binary_custom(b->grad, a->val, b->val, cur->grad,
                                             [](float p, float, float dC) {
                                               float pi = p > 0.0f ? p : 1e-12f;
                                               return -std::log(pi) * dC;
                                             });
      }
      break;
    }
    case OP_BCE: {
      if (a && a->grad) { // a = pred
        tensor_accumulate_grad_binary_custom(
            a->grad, a->val, b->val, cur->grad, [](float p, float t, float dC) {
              p = std::fmax(std::fmin(p, 1.0f - 1e-12f), 1e-12f);
              return (-t / p + (1.0f - t) / (1.0f - p)) * dC;
            });
      }
      break;
    }
    }
  }
}

} // namespace gradientcore
