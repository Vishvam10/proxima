# ================================
# Proxima Makefile
# ================================

BUILD_DIR := build
PYTHON_DIR := python
PYTHON := python3

.PHONY: all build test bench pybench clean rebuild help

all: build

# --------------------------------
# Build (configure only once)
# --------------------------------
build:
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "Configuring project..."; \
		mkdir -p $(BUILD_DIR); \
		cd $(BUILD_DIR) && cmake ..; \
	fi
	@echo "Building project..."
	@cmake --build $(BUILD_DIR) -j

# --------------------------------
# Run tests
# --------------------------------
test: build
	@cd $(BUILD_DIR) && ctest --output-on-failure

# --------------------------------
# Run C++ benchmark
# --------------------------------
bench: build
	@./$(BUILD_DIR)/bench

# --------------------------------
# Python benchmark
# --------------------------------
pybench:
	@cd $(PYTHON_DIR) && $(PYTHON) bench.py

# --------------------------------
# Clean
# --------------------------------
clean:
	rm -rf $(BUILD_DIR)

# --------------------------------
# Rebuild
# --------------------------------
rebuild: clean build

help:
	@echo "make build    - Build project"
	@echo "make test     - Run tests"
	@echo "make bench    - Run C++ benchmark"
	@echo "make pybench  - Run Python benchmark"
	@echo "make clean    - Clean build"