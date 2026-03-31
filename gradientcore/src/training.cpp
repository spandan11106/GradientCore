#include "include/gradientcore/training.hpp"
#include "include/gradientcore/graph.hpp"
#include "include/gradientcore/base/arena.hpp"
#include "include/gradientcore/base/prng.hpp"

#include <cstdio>
#include <cstring>

namespace gradientcore {

void model_train(model_context *model,
                 const model_training_desc *training_desc) {
  matrix *train_images = training_desc->train_images;
  matrix *train_labels = training_desc->train_labels;
  matrix *test_images = training_desc->test_images;
  matrix *test_labels = training_desc->test_labels;

  uint32_t num_examples = train_images->rows;
  uint32_t input_size = train_images->cols;
  uint32_t output_size = train_labels->cols;
  uint32_t num_test = test_images->rows;

  uint32_t num_batches = num_examples / training_desc->batch_size;

  ArenaTemp scratch = scratch_get(NULL, 0);

  uint32_t *training_order =
      scratch.arena->push_array<uint32_t>(num_examples, true);
  for (uint32_t i = 0; i < num_examples; i++) {
    training_order[i] = i;
  }

  for (uint32_t epoch = 0; epoch < training_desc->epochs; epoch++) {
    for (uint32_t i = 0; i < num_examples; i++) {
      uint32_t a = prng::rand() % num_examples;
      uint32_t b = prng::rand() % num_examples;

      uint32_t tmp = training_order[b];
      training_order[b] = training_order[a];
      training_order[a] = tmp;
    }

    for (uint32_t batch = 0; batch < num_batches; batch++) {
      for (uint32_t i = 0; i < model->cost_prog.size; i++) {
        model_var *cur = model->cost_prog.vars[i];

        if (cur->flags & MY_FLAG_PARAMETER) {
          mat_clear(cur->grad);
        }
      }
      float avg_cost = 0.0f;
      for (uint32_t i = 0; i < training_desc->batch_size; i++) {
        uint32_t order_index = batch * training_desc->batch_size + i;
        uint32_t index = training_order[order_index];

        memcpy(model->input->val->data, train_images->data + index * input_size,
               sizeof(float) * input_size);

        memcpy(model->desired_output->val->data,
               train_labels->data + index * output_size,
               sizeof(float) * output_size);

        model_prog_compute(&model->cost_prog);
        model_program_compute_grads(&model->cost_prog);

        avg_cost = mat_sum(model->cost->val);
      }
      avg_cost /= (float)training_desc->batch_size;

      for (uint32_t i = 0; i < model->cost_prog.size; i++) {
        model_var *cur = model->cost_prog.vars[i];

        if ((cur->flags & MY_FLAG_PARAMETER) != MY_FLAG_PARAMETER) {
          continue;
        }

        mat_scale(cur->grad,
                  training_desc->learning_rate / training_desc->batch_size);
        mat_sub(cur->val, cur->val, cur->grad);
      }

      printf("Epoch %2d / %2d, Batch %4d / %4d, Avarage Cost: %.4f\r",
             epoch + 1, training_desc->epochs, batch + 1, num_batches,
             avg_cost);
    }
    printf("\n");

    uint32_t num_correct = 0;
    float avg_cost = 0;
    for (uint32_t i = 0; i < num_test; i++) {
      memcpy(model->input->val->data, test_images->data + i * input_size,
             sizeof(float) * input_size);

      memcpy(model->desired_output->val->data,
             test_labels->data + i * output_size, sizeof(float) * output_size);

      model_prog_compute(&model->cost_prog);
      avg_cost += mat_sum(model->cost->val);
      num_correct += mat_argmax(model->output->val) ==
                     mat_argmax(model->desired_output->val);
    }

    avg_cost /= (float)num_test;
    printf("Test Completed : Accuracy %5d / %d (%1f%%), Avarage Cost: %.4f\n",
           num_correct, num_test, (float)num_correct / num_test * 100.0f,
           avg_cost);
  }
}

} // namespace gradientcore
