BUILD_DIR := build

RUN_DIR ?= $(shell date +%d-%m-%Y-%H-%M)

RESULTS_DIR := benchmarks/results

# Python used for building/testing distributable wheels.
# Development commands continue to use the project's uv environment.
PYTHON_VERSION ?= 3.13

PYTHON := uv run python
CMAKE := cmake
CLANG_FORMAT := clang-format
CLANG_TIDY := clang-tidy

CMAKE_FLAGS := \
	-DPROXIMA_BUILD_TESTS=ON \
	-DPROXIMA_BUILD_BENCHMARKS=ON


# --------------------------------------------------
# Default
# --------------------------------------------------

.PHONY: all

all: build


# --------------------------------------------------
# Setup
# --------------------------------------------------

.PHONY: setup

setup:
	./scripts/setup.sh


# --------------------------------------------------
# C++ Build
# --------------------------------------------------

.PHONY: configure

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)


.PHONY: build

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel


.PHONY: rebuild

rebuild:
	rm -rf $(BUILD_DIR)
	$(MAKE) build


.PHONY: clean

clean:
	rm -rf $(BUILD_DIR)


# --------------------------------------------------
# C++ Tests
# --------------------------------------------------

.PHONY: test

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure


# --------------------------------------------------
# Python Environment / Package
# --------------------------------------------------

.PHONY: python

python:
	uv pip install -e ".[test,benchmark]"


.PHONY: python-test

python-test: python
	uv run pytest tests/python -v


.PHONY: python-all-test

python-all-test: python
	uv run pytest -v


# --------------------------------------------------
# Python Package
# --------------------------------------------------

.PHONY: package

package:
	rm -rf dist
	uv build --python $(PYTHON_VERSION)


.PHONY: wheel

wheel:
	rm -rf dist
	uv build --wheel --python $(PYTHON_VERSION)


.PHONY: sdist

sdist:
	rm -rf dist
	uv build --sdist


.PHONY: package-check

package-check: package
	uvx twine check dist/*


.PHONY: package-test

package-test: package
	@set -e; \
	rm -rf /tmp/proxima-package-test; \
	uv venv \
		--python $(PYTHON_VERSION) \
		/tmp/proxima-package-test; \
	uv pip install \
		--python /tmp/proxima-package-test/bin/python \
		dist/*.whl; \
	uv pip install \
		--python /tmp/proxima-package-test/bin/python \
		pytest numpy; \
	cd /tmp/proxima-package-test; \
	./bin/python -c \
		"import proxima; print('Proxima imported from:', proxima.__file__)"; \
	./bin/python -m pytest \
		$(CURDIR)/tests/python \
		-v

# --------------------------------------------------
# C++ Benchmarks
# --------------------------------------------------

.PHONY: cppbench

cppbench: build
	mkdir -p $(RESULTS_DIR)/$(RUN_DIR)
	./$(BUILD_DIR)/bench_proxima $(RESULTS_DIR)/$(RUN_DIR)


# --------------------------------------------------
# Python Benchmarks
# --------------------------------------------------

.PHONY: pybindings-bench

pybindings-bench: python
	mkdir -p $(RESULTS_DIR)/$(RUN_DIR)
	$(PYTHON) benchmarks/bench_proxima_pybinding.py \
		$(RESULTS_DIR)/$(RUN_DIR)


.PHONY: pybench

pybench: python
	mkdir -p $(RESULTS_DIR)/$(RUN_DIR)
	$(PYTHON) benchmarks/bench_hnswlib.py \
		$(RESULTS_DIR)/$(RUN_DIR)


# --------------------------------------------------
# Full Benchmarks
# --------------------------------------------------

.PHONY: bench

bench:
	$(MAKE) build
	$(MAKE) python
	mkdir -p $(RESULTS_DIR)/$(RUN_DIR)
	$(MAKE) cppbench RUN_DIR=$(RUN_DIR)
	$(MAKE) pybindings-bench RUN_DIR=$(RUN_DIR)
	$(MAKE) pybench RUN_DIR=$(RUN_DIR)
	$(PYTHON) benchmarks/compare.py \
		$(RESULTS_DIR)/$(RUN_DIR)
	$(PYTHON) benchmarks/plot.py \
		$(RESULTS_DIR)/$(RUN_DIR)


# --------------------------------------------------
# Plots / Reports
# --------------------------------------------------

.PHONY: plot

plot:
	$(PYTHON) benchmarks/plot.py \
		$(RESULTS_DIR)/$(RUN_DIR)


.PHONY: compare

compare:
	$(PYTHON) benchmarks/compare.py \
		$(RESULTS_DIR)/$(RUN_DIR)


# --------------------------------------------------
# Formatting
# --------------------------------------------------

CPP_FILES := $(shell find src include tests/cpp benchmarks \
	-name '*.cpp' -o -name '*.h' -o -name '*.hpp')


.PHONY: format

format:
	$(CLANG_FORMAT) -i $(CPP_FILES)


.PHONY: format-check

format-check:
	@$(CLANG_FORMAT) --dry-run --Werror $(CPP_FILES)


# --------------------------------------------------
# Static Analysis
# --------------------------------------------------

.PHONY: lint

lint: build
	$(CLANG_TIDY) \
		-p $(BUILD_DIR) \
		$$(find src -name '*.cpp')


.PHONY: lint-fix

lint-fix: build
	$(CLANG_TIDY) \
		-p $(BUILD_DIR) \
		--fix \
		$$(find src -name '*.cpp')


# --------------------------------------------------
# CI
# --------------------------------------------------

.PHONY: ci

ci:
	$(MAKE) format-check
	$(MAKE) build
	$(MAKE) test
	$(MAKE) python-test
	$(MAKE) package-test


# --------------------------------------------------
# Release
# --------------------------------------------------

.PHONY: release-check

release-check:
	@echo "Running release checks..."
	$(MAKE) format-check
	$(MAKE) build
	$(MAKE) test
	$(MAKE) python-test
	$(MAKE) package-test
	$(MAKE) package-check
	@echo "Release checks passed."


# --------------------------------------------------
# Help
# --------------------------------------------------

.PHONY: help

help:
	@echo ""
	@echo "Proxima"
	@echo ""
	@echo "Build:"
	@echo "  make setup                 Install development tools"
	@echo "  make configure             Configure CMake"
	@echo "  make build                 Build C++ project"
	@echo "  make rebuild               Clean rebuild"
	@echo "  make clean                 Remove build directory"
	@echo ""
	@echo "Tests:"
	@echo "  make test                  Run C++ tests"
	@echo "  make python-test           Run Python tests"
	@echo "  make python-all-test       Run all Python tests"
	@echo "  make ci                    Run full CI checks"
	@echo ""
	@echo "Python:"
	@echo "  make python                Install Python package/dependencies"
	@echo "  make package               Build wheel + sdist"
	@echo "  make wheel                 Build wheel only"
	@echo "  make sdist                 Build source distribution"
	@echo "  make package-check         Validate distributions"
	@echo "  make package-test          Test actual wheel in clean venv"
	@echo ""
	@echo "Packaging Python version: $(PYTHON_VERSION)"
	@echo "  Override with: make package PYTHON_VERSION=3.12"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make cppbench              Run C++ benchmark"
	@echo "  make pybindings-bench      Run Python bindings benchmark"
	@echo "  make pybench               Run hnswlib benchmark"
	@echo "  make bench                 Run complete benchmark suite"
	@echo "  make compare               Generate comparison"
	@echo "  make plot                  Generate plots"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format                Format C++ code"
	@echo "  make format-check          Check formatting"
	@echo "  make lint                  Run clang-tidy"
	@echo "  make lint-fix              Run clang-tidy with fixes"
	@echo ""
	@echo "Release:"
	@echo "  make release-check         Run all release checks"
	@echo ""
	@echo "Benchmark example:"
	@echo "  make bench RUN_DIR=my-run"
	@echo ""
