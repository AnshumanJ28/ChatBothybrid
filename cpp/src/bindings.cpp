#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lstm_cell.h"
#include "embedding_search.h"
#include "dense_layer.h"
#include "text_sanitizer.h"
#include "attention_pooling.h"
#include "math_evaluator.h"
#include "cognitive_engine.h"

namespace py = pybind11;
using namespace minibrain;

PYBIND11_MODULE(minibrain_cpp, m) {
    m.doc() = "Hand-written C++ LSTM cell, embedding similarity search, "
              "MathEvaluator, CognitiveEngine, and TextSanitizer algorithm bound to Python via pybind11.";

    py::class_<LSTMCellWeights>(m, "LSTMCellWeights")
        .def(py::init<>())
        .def_readwrite("weight_ih", &LSTMCellWeights::weight_ih)
        .def_readwrite("weight_hh", &LSTMCellWeights::weight_hh)
        .def_readwrite("bias_ih", &LSTMCellWeights::bias_ih)
        .def_readwrite("bias_hh", &LSTMCellWeights::bias_hh)
        .def_readwrite("input_size", &LSTMCellWeights::input_size)
        .def_readwrite("hidden_size", &LSTMCellWeights::hidden_size);

    py::class_<LSTMCellState>(m, "LSTMCellState")
        .def(py::init<>())
        .def_readwrite("h", &LSTMCellState::h)
        .def_readwrite("c", &LSTMCellState::c);

    py::class_<LSTMCell>(m, "LSTMCell")
        .def(py::init<LSTMCellWeights>())
        .def("forward", &LSTMCell::forward,
             py::arg("x"), py::arg("h_prev") = std::vector<double>{},
             py::arg("c_prev") = std::vector<double>{},
             "Run one timestep. h_prev/c_prev default to zeros.")
        .def_property_readonly("input_size", &LSTMCell::input_size)
        .def_property_readonly("hidden_size", &LSTMCell::hidden_size);

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("index", &SearchResult::index)
        .def_readonly("score", &SearchResult::score);

    py::class_<EmbeddingIndex>(m, "EmbeddingIndex")
        .def(py::init<int>(), py::arg("dim"))
        .def("add", &EmbeddingIndex::add, py::arg("vec"), py::arg("doc_id"))
        .def("search", &EmbeddingIndex::search, py::arg("query"), py::arg("top_k"))
        .def("doc_id", &EmbeddingIndex::doc_id, py::arg("index"))
        .def_property_readonly("size", &EmbeddingIndex::size)
        .def_property_readonly("dim", &EmbeddingIndex::dim);

    py::class_<StackedLSTMState>(m, "StackedLSTMState")
        .def(py::init<>())
        .def_readwrite("h", &StackedLSTMState::h)
        .def_readwrite("c", &StackedLSTMState::c);

    py::class_<DeepLSTM>(m, "DeepLSTM")
        .def(py::init<std::vector<LSTMCellWeights>>())
        .def("forward", &DeepLSTM::forward,
             py::arg("x"), py::arg("h_prev") = std::vector<std::vector<double>>{},
             py::arg("c_prev") = std::vector<std::vector<double>>{},
             "Run one timestep of stacked LSTMs. h_prev/c_prev default to empty (zeros).")
        .def_property_readonly("num_layers", &DeepLSTM::num_layers);

    py::class_<DenseLayerWeights>(m, "DenseLayerWeights")
        .def(py::init<>())
        .def_readwrite("weights", &DenseLayerWeights::weights)
        .def_readwrite("bias", &DenseLayerWeights::bias)
        .def_readwrite("input_size", &DenseLayerWeights::input_size)
        .def_readwrite("output_size", &DenseLayerWeights::output_size);

    py::class_<DenseLayer>(m, "DenseLayer")
        .def(py::init<DenseLayerWeights>())
        .def("forward", &DenseLayer::forward, py::arg("x"),
             "Run forward pass with ReLU activation: out = max(0.0, W * x + b)")
        .def_property_readonly("input_size", &DenseLayer::input_size)
        .def_property_readonly("output_size", &DenseLayer::output_size);

    py::class_<TextSanitizer>(m, "TextSanitizer")
        .def_static("remove_citations", &TextSanitizer::remove_citations, py::arg("text"))
        .def_static("remove_unit_tables", &TextSanitizer::remove_unit_tables, py::arg("text"))
        .def_static("remove_nav_junk", &TextSanitizer::remove_nav_junk, py::arg("text"))
        .def_static("sanitize", &TextSanitizer::sanitize, py::arg("raw_text"));

    py::class_<AttentionPooling>(m, "AttentionPooling")
        .def_static("pool", &AttentionPooling::pool, py::arg("token_embeds"));

    py::class_<MathResult>(m, "MathResult")
        .def_readonly("is_math", &MathResult::is_math)
        .def_readonly("value", &MathResult::value)
        .def_readonly("formatted_result", &MathResult::formatted_result)
        .def_readonly("error", &MathResult::error);

    py::class_<MathEvaluator>(m, "MathEvaluator")
        .def_static("is_math_query", &MathEvaluator::is_math_query, py::arg("text"))
        .def_static("evaluate", &MathEvaluator::evaluate, py::arg("query"));

    py::class_<CognitiveTrace>(m, "CognitiveTrace")
        .def_readonly("confidence", &CognitiveTrace::confidence)
        .def_readonly("primary_subsystem", &CognitiveTrace::primary_subsystem)
        .def_readonly("thoughts", &CognitiveTrace::thoughts);

    py::class_<CognitiveEngine>(m, "CognitiveEngine")
        .def_static("process", &CognitiveEngine::process,
                    py::arg("query"), py::arg("tokens"), py::arg("lstm_norms"));
}
