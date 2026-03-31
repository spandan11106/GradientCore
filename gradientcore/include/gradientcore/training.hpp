#pragma once

#include "matrix.hpp"
#include "model.hpp"

namespace gradientcore {

struct model_training_desc {
  matrix *train_images;
  matrix *train_labels;
  matrix *test_images;
  matrix *test_labels;

  uint32_t epochs;
  uint32_t batch_size;
  float learning_rate;
};

// Training
void model_train(model_context *model,
                 const model_training_desc *training_desc);

} // namespace gradientcore
