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

SRC := $(shell find src -name "*.cpp" ! -name "scratchpad.cpp")
CPP_FILES := $(shell find src -name "*.cpp" -o -name "*.hpp" -o -name "*.h")

.PHONY: all setup dev build rebuild clean test scratchpad venv cppbench pybench bench plot format format-check lint lint-fix help

# ----------------------------
all: build

# ----------------------------
setup:
	@bash scripts/setup.sh
	@$(MAKE) $(VENV_PYTHON)

# ----------------------------
dev: format lint test

# ----------------------------
build:
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "Configuring project..."; \
		mkdir -p $(BUILD_DIR); \
		cd $(BUILD_DIR) && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..; \
	fi
	@echo "Building project..."
	@cmake --build $(BUILD_DIR) -j$(NPROC)
	@ln -sf $(BUILD_DIR)/compile_commands.json .

# ----------------------------
rebuild: clean build

# ----------------------------
scratchpad: clean
	@echo "Compiling and running scratchpad..."
	@mkdir -p $(BUILD_DIR)
	@clang++ -std=c++17 -g -O1 \
	-fsanitize=address,undefined \
	-fsanitize-address-use-after-scope \
	-fno-omit-frame-pointer \
	-Isrc $(SRC) src/scratchpad.cpp -o $(BUILD_DIR)/scratchpad
	@$(BUILD_DIR)/scratchpad

# ----------------------------
test: clean build
	@cd $(BUILD_DIR) && ctest --output-on-failure

# ----------------------------
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
	    echo "Setting up Python virtual environment..."; \
	    $(PYTHON) -m venv $(VENV_DIR); \
	    echo "Installing benchmark dependencies..."; \
	    $(VENV_PYTHON) -m pip install --upgrade pip; \
	    $(VENV_PYTHON) -m pip install -r $(BENCH_DIR)/requirements.txt; \
	else \
	    echo "Virtual environment already exists."; \
	fi

cppbench: build
	@mkdir -p $(BENCH_DIR)/results/$(RUN_DIR)
	@./$(BUILD_DIR)/bench $(BENCH_DIR)/results/$(RUN_DIR)

pybench: venv
	@mkdir -p $(BENCH_DIR)/results/$(RUN_DIR)
	@$(VENV_PYTHON) $(BENCH_DIR)/bench.py $(RUN_DIR)

# ----------------------------
bench: build venv
	@RUN_DIR=$$(date +"%d-%m-%Y-%H-%M"); \
	mkdir -p $(BENCH_DIR)/results/$$RUN_DIR/plots; \
	echo "Results directory: $$RUN_DIR"; \
	echo "Running C++ benchmarks..."; \
	./$(BUILD_DIR)/bench $(BENCH_DIR)/results/$$RUN_DIR; \
	echo "Running Python benchmarks..."; \
	$(VENV_PYTHON) $(BENCH_DIR)/bench.py $$RUN_DIR; \
	echo "Comparing results..."; \
	$(VENV_PYTHON) $(BENCH_DIR)/compare.py $$RUN_DIR; \
	echo "Generating plots..."; \
	$(VENV_PYTHON) $(BENCH_DIR)/plot.py $$RUN_DIR; \
	echo "Done. Results saved to $(BENCH_DIR)/results/$$RUN_DIR/"

# ----------------------------
plot: venv
	@LATEST=$$(ls -dt $(BENCH_DIR)/results/*/ 2>/dev/null | head -1) && \
	if [ -z "$$LATEST" ]; then \
		echo "No results found. Run 'make bench' first."; \
		exit 1; \
	fi && \
	RUN_DIR=$$(basename $$LATEST) && \
	$(VENV_PYTHON) $(BENCH_DIR)/plot.py $$RUN_DIR

# ----------------------------
format:
	@echo "Formatting C++ code..."
	@$(CLANG_FORMAT) -i $(CPP_FILES)

format-check:
	@echo "Checking formatting..."
	@$(CLANG_FORMAT) --dry-run --Werror $(CPP_FILES)

# ----------------------------
lint: build
	@echo "Running clang-tidy (parallel)..."
	@printf "%s\n" $(SRC) | xargs -P $(NPROC) -I{} $(CLANG_TIDY) -p $(COMPILE_DB) {}

lint-fix: build
	@echo "Running clang-tidy with fixes..."
	@printf "%s\n" $(SRC) | xargs -P $(NPROC) -I{} $(CLANG_TIDY) -p $(COMPILE_DB) --fix {}

# ----------------------------
clean:
	rm -rf $(BUILD_DIR)
	rm -f compile_commands.json

# ----------------------------
help:
	@echo "Build & Development"
	@echo "-------------------"
	@echo "make setup        - Setup dev environment"
	@echo "make build        - Build project (CMake)"
	@echo "make rebuild      - Clean and rebuild"
	@echo "make dev          - Format + lint + test"
	@echo ""
	@echo "Testing"
	@echo "-------"
	@echo "make test         - Run tests"
	@echo "make scratchpad   - Compile and run scratchpad (with sanitizers)"
	@echo ""
	@echo "Benchmarks"
	@echo "----------"
	@echo "make cppbench     - Run C++ benchmark (sanitizer-free)"
	@echo "make pybench      - Run Python benchmark"
	@echo "make bench        - Run full benchmark suite"
	@echo "make plot         - Plot latest results"
	@echo ""
	@echo "Code Quality"
	@echo "------------"
	@echo "make format       - Format code"
	@echo "make format-check - Check formatting"
	@echo "make lint         - Run clang-tidy"
	@echo "make lint-fix     - Run clang-tidy and apply fixes"
	@echo ""
	@echo "Maintenance"
	@echo "-----------"
	@echo "make clean        - Remove build artifacts"