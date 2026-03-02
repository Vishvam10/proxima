# ================================
# Proxima Makefile
# ================================

BUILD_DIR := build
BENCH_DIR := benchmarks
PYTHON := python3
VENV_DIR := $(BENCH_DIR)/.venv
VENV_PYTHON := $(VENV_DIR)/bin/python

# Automatically find all .cpp files inside src
SRC := $(shell find src -name "*.cpp")

.PHONY: all build test bench cppbench pybench clean rebuild help scratchpad

all: build

# --------------------------------
# Run scratchpad
# --------------------------------
scratchpad:
	@echo "Compiling and running scratchpad..."
	@mkdir -p $(BUILD_DIR)
	@clang++ -std=c++17 -O3 -Isrc $(SRC) -o $(BUILD_DIR)/scratchpad
	@$(BUILD_DIR)/scratchpad

# --------------------------------
# Build (CMake)
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
cppbench: build
	@./$(BUILD_DIR)/bench

# --------------------------------
# Python benchmark (with venv)
# --------------------------------
$(VENV_PYTHON): $(BENCH_DIR)/requirements.txt
	@echo "Creating virtual environment in $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing Python benchmark dependencies..."
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PYTHON) -m pip install -r $(BENCH_DIR)/requirements.txt

pybench: $(VENV_PYTHON)
	@$(VENV_PYTHON) $(BENCH_DIR)/bench.py

# --------------------------------
# Run both benchmarks and compare
# --------------------------------
bench: cppbench pybench
	@$(VENV_PYTHON) $(BENCH_DIR)/compare.py

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
	@echo "make build      - Build project (CMake)"
	@echo "make test       - Run tests"
	@echo "make cppbench   - Run C++ benchmark"
	@echo "make pybench    - Run Python benchmark (in venv)"
	@echo "make bench      - Run both benchmarks and compare"
	@echo "make scratchpad - Compile and run scratchpad directly"
	@echo "make clean      - Clean build"