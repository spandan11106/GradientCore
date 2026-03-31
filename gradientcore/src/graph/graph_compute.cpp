#include "../../include/gradientcore/graph.hpp"
#include "../../include/gradientcore/matrix.hpp"

namespace gradientcore {

void model_prog_compute(model_program *prog) {
  for (uint32_t i = 0; i < prog->size; i++) {
    model_var *cur = prog->vars[i];

    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    switch (cur->op) {
    case MV_OP_NULL:
    case MV_OP_CREATE:
      break;

    case _MV_OP_UNARY_START:
      break;

    case MV_OP_RELU: {
      mat_relu(cur->val, a->val);
    } break;

    case MV_OP_SOFTMAX: {
      mat_softmax(cur->val, a->val);
    } break;

    case _MV_OP_BINARY_START:
      break;

    case MV_OP_ADD: {
      mat_add(cur->val, a->val, b->val);
    } break;

    case MV_OP_SUB: {
      mat_sub(cur->val, a->val, b->val);
    } break;

    case MV_OP_MATMUL: {
      mat_mul(cur->val, a->val, b->val, true, false, false);
    } break;

    case MV_OP_CROSS_ENTROPY: {
      mat_cross_entropy(cur->val, a->val, b->val);
    } break;
    }
  }
}

void model_program_compute_grads(model_program *prog) {
  for (uint32_t i = 0; i < prog->size; i++) {
    model_var *cur = prog->vars[i];

    if ((cur->flags & MY_FLAG_REQUIRES_GRAD) != MY_FLAG_REQUIRES_GRAD) {
      continue;
    }

    if (cur->flags & MY_FLAG_PARAMETER) {
      continue;
    }

    mat_clear(cur->grad);
  }

  mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

  for (int64_t i = (int64_t)prog->size - 1; i >= 0; i--) {
    model_var *cur = prog->vars[i];

    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    uint32_t num_inputs = MY_NUM_INPUTS(cur->op);

    if (num_inputs == 1 &&
        (a->flags & MY_FLAG_REQUIRES_GRAD) != MY_FLAG_REQUIRES_GRAD) {
      continue;
    }

    if (num_inputs == 2 &&
        (a->flags & MY_FLAG_REQUIRES_GRAD) != MY_FLAG_REQUIRES_GRAD &&
        (b->flags & MY_FLAG_REQUIRES_GRAD) != MY_FLAG_REQUIRES_GRAD) {
      continue;
    }

    switch (cur->op) {
    case MV_OP_NULL:
    case MV_OP_CREATE:
      break;

    case _MV_OP_UNARY_START:
      break;

    case MV_OP_RELU: {
      mat_relu_add_grad(a->grad, a->val, cur->grad);
    } break;

    case MV_OP_SOFTMAX: {
      mat_softmax_add_grad(a->grad, cur->val, cur->grad);
    } break;

    case _MV_OP_BINARY_START:
      break;

    case MV_OP_ADD: {
      if (a->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_add(a->grad, a->grad, cur->grad);
      }
      if (b->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_add(b->grad, b->grad, cur->grad);
      }
    } break;

    case MV_OP_SUB: {
      if (a->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_add(a->grad, a->grad, cur->grad);
      }
      if (b->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_sub(b->grad, b->grad, cur->grad);
      }

    } break;

    case MV_OP_MATMUL: {
      if (a->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
      }
      if (b->flags & MY_FLAG_REQUIRES_GRAD) {
        mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
      }
    } break;

    case MV_OP_CROSS_ENTROPY: {
      model_var *p = a;
      model_var *q = b;

      mat_cross_entorpy_add_grad(p->grad, q->grad, p->val, q->val, cur->grad);
    } break;
    }
  }
}

} // namespace gradientcore
