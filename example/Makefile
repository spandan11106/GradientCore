CXX = g++
LD = ld
CXXFLAGS = -I../gradientcore -std=c++17 -Wall -Wextra
TARGET = mnist_example
INFERENCE_TARGET = inference_example
BUILD_DIR = .
MONOLITHIC = gradientcore.o

# All core object files EXCEPT main.o and inference.o
CORE_OBJECTS = $(BUILD_DIR)/mnist.o \
          $(BUILD_DIR)/matrix_ops.o \
          $(BUILD_DIR)/matrix_arithmetic.o \
          $(BUILD_DIR)/matrix_multiply.o \
          $(BUILD_DIR)/matrix_activations.o \
          $(BUILD_DIR)/model.o \
          $(BUILD_DIR)/operators_unary.o \
          $(BUILD_DIR)/operators_binary.o \
          $(BUILD_DIR)/graph_topological.o \
          $(BUILD_DIR)/graph_compute.o \
          $(BUILD_DIR)/graph_compile.o \
          $(BUILD_DIR)/training_sgd.o \
          $(BUILD_DIR)/arena.o \
          $(BUILD_DIR)/prng.o \
          $(BUILD_DIR)/platform.o

all: $(TARGET) $(INFERENCE_TARGET)

# 1. Build the monolithic core library first
$(MONOLITHIC): $(CORE_OBJECTS)
	@$(LD) -r $^ -o $@
	@echo "✓ Linked Core: $(MONOLITHIC)"

# 2. Link the training app
$(TARGET): $(MONOLITHIC) $(BUILD_DIR)/main.o
	@$(CXX) $(CXXFLAGS) $(BUILD_DIR)/main.o $(MONOLITHIC) -o $@
	@echo "✓ Built Training App: $(TARGET)"

# 3. Link the inference app
$(INFERENCE_TARGET): $(MONOLITHIC) $(BUILD_DIR)/inference.o
	@$(CXX) $(CXXFLAGS) $(BUILD_DIR)/inference.o $(MONOLITHIC) -o $@
	@echo "✓ Built Inference App: $(INFERENCE_TARGET)"

# Example sources
$(BUILD_DIR)/main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/inference.o: inference.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist.o: mnist.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Library matrix operations
$(BUILD_DIR)/matrix_ops.o: ../gradientcore/src/matrix/matrix_ops.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/matrix_arithmetic.o: ../gradientcore/src/matrix/matrix_arithmetic.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/matrix_multiply.o: ../gradientcore/src/matrix/matrix_multiply.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/matrix_activations.o: ../gradientcore/src/matrix/matrix_activations.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Library model management
$(BUILD_DIR)/model.o: ../gradientcore/src/model.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Library operators
$(BUILD_DIR)/operators_unary.o: ../gradientcore/src/operators/operators_unary.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/operators_binary.o: ../gradientcore/src/operators/operators_binary.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Library graph
$(BUILD_DIR)/graph_topological.o: ../gradientcore/src/graph/graph_topological.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/graph_compute.o: ../gradientcore/src/graph/graph_compute.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/graph_compile.o: ../gradientcore/src/graph/graph_compile.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Library training
$(BUILD_DIR)/training_sgd.o: ../gradientcore/src/training/training_sgd.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Library base
$(BUILD_DIR)/arena.o: ../gradientcore/src/base/arena.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/prng.o: ../gradientcore/src/base/prng.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Library platform
$(BUILD_DIR)/platform.o: ../gradientcore/src/platform/platform.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@rm -f $(CORE_OBJECTS) $(BUILD_DIR)/main.o $(BUILD_DIR)/inference.o $(MONOLITHIC) $(TARGET) $(INFERENCE_TARGET)
	@echo "✓ Cleaned up all build artifacts"

run: $(TARGET)
	./$(TARGET)
	@$(MAKE) clean

run-inference: $(INFERENCE_TARGET)
	./$(INFERENCE_TARGET)
	@$(MAKE) clean

.PHONY: all clean run run-inference