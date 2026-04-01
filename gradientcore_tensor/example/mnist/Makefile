CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

ROOT     := ../../..
TENSOR   := $(ROOT)/gradientcore_tensor
BASE     := $(ROOT)/gradientcore

TARGET_TRAIN := mnist_train
TARGET_INFER := mnist_inference

TENSOR_SRC := \
	$(TENSOR)/src/core/arena.cpp \
	$(TENSOR)/src/core/tensor.cpp \
	$(TENSOR)/src/platform/platform.cpp \
	$(TENSOR)/src/autograd/autograd.cpp \
	$(TENSOR)/src/autograd/engine.cpp \
	$(TENSOR)/src/ops/ops_utils.cpp \
	$(TENSOR)/src/ops/add.cpp \
	$(TENSOR)/src/ops/sub.cpp \
	$(TENSOR)/src/ops/mul.cpp \
	$(TENSOR)/src/ops/div.cpp \
	$(TENSOR)/src/ops/matmul.cpp \
	$(TENSOR)/src/ops/neg.cpp \
	$(TENSOR)/src/ops/exp.cpp \
	$(TENSOR)/src/ops/log.cpp \
	$(TENSOR)/src/ops/pow.cpp \
	$(TENSOR)/src/ops/scalar_ops.cpp \
	$(TENSOR)/src/ops/sum.cpp \
	$(TENSOR)/src/ops/mean.cpp \
	$(TENSOR)/src/ops/relu.cpp \
	$(TENSOR)/src/ops/leaky_relu.cpp \
	$(TENSOR)/src/ops/sigmoid.cpp \
	$(TENSOR)/src/ops/tanh.cpp \
	$(TENSOR)/src/ops/gelu.cpp \
	$(TENSOR)/src/ops/silu.cpp \
	$(TENSOR)/src/ops/softmax.cpp \
	$(TENSOR)/src/ops/mse.cpp \
	$(TENSOR)/src/ops/l1_loss.cpp \
	$(TENSOR)/src/ops/cross_entropy.cpp \
	$(TENSOR)/src/ops/bce.cpp \
	$(TENSOR)/src/nn/init.cpp \
	$(TENSOR)/src/nn/linear.cpp \
	$(TENSOR)/src/optim/sgd.cpp \
	$(TENSOR)/src/optim/adam.cpp \
	$(TENSOR)/src/optim/utils.cpp \
	$(TENSOR)/src/serialize/serialize.cpp

BASE_SRC := \
	$(BASE)/src/base/prng.cpp

SRCS_TRAIN := mnist_train.cpp $(TENSOR_SRC) $(BASE_SRC)
SRCS_INFER := mnist_inference.cpp $(TENSOR_SRC) $(BASE_SRC)

all: build

build: $(TARGET_TRAIN) $(TARGET_INFER)

$(TARGET_TRAIN): $(SRCS_TRAIN)
	@echo "Compiling MNIST trainer..."
	@$(CXX) $(CXXFLAGS) -o $@ $(SRCS_TRAIN)
	@echo "Train Build complete."

$(TARGET_INFER): $(SRCS_INFER)
	@echo "Compiling MNIST inference..."
	@$(CXX) $(CXXFLAGS) -o $@ $(SRCS_INFER)
	@echo "Inference Build complete."

run: $(TARGET_TRAIN)
	@./$(TARGET_TRAIN)

run-inference: $(TARGET_INFER)
	@./$(TARGET_INFER)

clean:
	@rm -f $(TARGET_TRAIN) $(TARGET_INFER) 

.PHONY: all build run run-inference clean
