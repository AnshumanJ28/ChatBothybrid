cmake_minimum_required(VERSION 3.15)
project(minibrain_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(minibrain_cpp
    src/lstm_cell.cpp
    src/embedding_search.cpp
    src/dense_layer.cpp
    src/text_sanitizer.cpp
    src/math_evaluator.cpp
    src/cognitive_engine.cpp
    src/bindings.cpp
)

target_include_directories(minibrain_cpp PRIVATE src)

# Drop the built extension next to the orchestrator package so
# `import minibrain_cpp` works from python/orchestrator without
# manual copying.
set_target_properties(minibrain_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../python/orchestrator"
)
