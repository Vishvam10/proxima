# ================================
# Proxima Makefile
# ================================

BUILD_DIR := build
BENCH_DIR := benchmarks

PYTHON := python3
VENV_DIR := $(BENCH_DIR)/.venv
VENV_PYTHON := $(VENV_DIR)/bin/python

CLANG_FORMAT := clang-format
CLANG_TIDY := clang-tidy

COMPILE_DB := $(BUILD_DIR)

# Detect CPU cores for parallel jobs
NPROC := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Automatically discover sources
SRC := $(shell find src -name "*.cpp")
CPP_FILES := $(shell find src -name "*.cpp" -o -name "*.hpp" -o -name "*.h")

.PHONY: \
	all setup dev \
	build rebuild clean \
	test scratchpad \
	cppbench pybench bench plot \
	format format-check lint lint-fix \
	help

# --------------------------------
# Default target
# --------------------------------
all: build

# --------------------------------
# Setup development environment
# --------------------------------
setup:
	@bash scripts/setup.sh
	@$(MAKE) $(VENV_PYTHON)

# --------------------------------
# Dev workflow (recommended)
# --------------------------------
dev: format lint test

# --------------------------------
# Build (CMake)
# --------------------------------
build:
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "Configuring project..."; \
		mkdir -p $(BUILD_DIR); \
		cd $(BUILD_DIR) && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..; \
	fi
	@echo "Building project..."
	@cmake --build $(BUILD_DIR) -j$(NPROC)
	@ln -sf $(BUILD_DIR)/compile_commands.json .

# --------------------------------
# Rebuild
# --------------------------------
rebuild: clean build

# --------------------------------
# Run scratchpad
# --------------------------------
scratchpad:
	@echo "Compiling and running scratchpad..."
	@mkdir -p $(BUILD_DIR)
	@clang++ -std=c++17 -O3 -Isrc $(SRC) -o $(BUILD_DIR)/scratchpad
	@$(BUILD_DIR)/scratchpad

# --------------------------------
# Run tests
# --------------------------------
test: build
	@cd $(BUILD_DIR) && ctest --output-on-failure

# --------------------------------
# C++ benchmark
# --------------------------------
cppbench: build
	@./$(BUILD_DIR)/bench

# --------------------------------
# Python benchmark (venv)
# --------------------------------
$(VENV_PYTHON): $(BENCH_DIR)/requirements.txt
	@echo "Creating Python virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)

	@echo "Installing benchmark dependencies..."
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PYTHON) -m pip install -r $(BENCH_DIR)/requirements.txt

pybench: $(VENV_PYTHON)
	@$(VENV_PYTHON) $(BENCH_DIR)/bench.py

# --------------------------------
# Run full benchmark suite
# --------------------------------
bench: cppbench pybench
	@RESULTS_DIR=$$(date +"%d-%m-%Y-%H-%M-%S") && \
	$(VENV_PYTHON) $(BENCH_DIR)/compare.py $$RESULTS_DIR && \
	$(VENV_PYTHON) $(BENCH_DIR)/plot.py $$RESULTS_DIR

# --------------------------------
# Plot latest results
# --------------------------------
plot: $(VENV_PYTHON)
	@LATEST=$$(ls -t $(BENCH_DIR)/results/ | head -1) && \
	if [ -z "$$LATEST" ]; then \
		echo "No results found. Run 'make bench' first."; \
		exit 1; \
	fi && \
	$(VENV_PYTHON) $(BENCH_DIR)/plot.py $$LATEST

# --------------------------------
# Formatting
# --------------------------------
format:
	@echo "Formatting C++ code..."
	@$(CLANG_FORMAT) -i $(CPP_FILES)

format-check:
	@echo "Checking formatting..."
	@$(CLANG_FORMAT) --dry-run --Werror $(CPP_FILES)

# --------------------------------
# Linting
# --------------------------------
lint: build
	@echo "Running clang-tidy (parallel)..."
	@printf "%s\n" $(SRC) | xargs -P $(NPROC) -I{} $(CLANG_TIDY) {} -p $(COMPILE_DB)

lint-fix: build
	@echo "Running clang-tidy with fixes..."
	@printf "%s\n" $(SRC) | xargs -P $(NPROC) -I{} $(CLANG_TIDY) {} -p $(COMPILE_DB) -fix

# --------------------------------
# Clean
# --------------------------------
clean:
	rm -rf $(BUILD_DIR)
	rm -f compile_commands.json

# --------------------------------
# Help
# --------------------------------
help:
	@echo ""
	@echo "Build & Development"
	@echo "-------------------"
	@echo "make setup        - Setup development environment"
	@echo "make build        - Build project (CMake)"
	@echo "make rebuild      - Clean and rebuild"
	@echo "make dev          - Format + lint + test"
	@echo ""
	@echo "Testing"
	@echo "-------"
	@echo "make test         - Run tests"
	@echo "make scratchpad   - Compile and run scratchpad"
	@echo ""
	@echo "Benchmarks"
	@echo "----------"
	@echo "make cppbench     - Run C++ benchmark"
	@echo "make pybench      - Run Python benchmark"
	@echo "make bench        - Run full benchmark suite"
	@echo "make plot         - Plot latest results"
	@echo ""
	@echo "Code Quality"
	@echo "------------"
	@echo "make format       - Format code with clang-format"
	@echo "make format-check - Check formatting (CI)"
	@echo "make lint         - Run clang-tidy"
	@echo "make lint-fix     - Run clang-tidy and apply fixes"
	@echo ""
	@echo "Maintenance"
	@echo "-----------"
	@echo "make clean        - Remove build artifacts"