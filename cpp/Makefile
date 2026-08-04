# Builds minibrain_cpp*.so directly with g++ + pybind11, no CMake required.
# If you prefer CMake, a CMakeLists.txt is also provided (see README).

PYTHON      ?= python3
PYBIND_INC  := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())")
PY_INC      := $(shell $(PYTHON)-config --includes)
PY_EXT_SUF  := $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

CXX         ?= g++
CXXFLAGS    := -O3 -Wall -shared -std=c++17 -fPIC $(PY_INC) -I$(PYBIND_INC) -Isrc

SRC         := src/lstm_cell.cpp src/embedding_search.cpp src/dense_layer.cpp src/text_sanitizer.cpp src/math_evaluator.cpp src/cognitive_engine.cpp src/bindings.cpp
TARGET      := minibrain_cpp$(PY_EXT_SUF)
OUTDIR      := ../python/orchestrator

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)
	@mkdir -p $(OUTDIR)
	@cp $(TARGET) $(OUTDIR)/$(TARGET)
	@echo "Built $(TARGET) and copied to $(OUTDIR)/"

clean:
	rm -f minibrain_cpp*.so $(OUTDIR)/minibrain_cpp*.so

test: all
	$(PYTHON) tests/test_bindings_smoke.py
