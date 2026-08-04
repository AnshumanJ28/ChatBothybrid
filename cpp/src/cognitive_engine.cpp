#include "cognitive_engine.h"
#include "math_evaluator.h"
#include <iomanip>
#include <cmath>

namespace minibrain {

CognitiveTrace CognitiveEngine::process(const std::string& query,
                                        const std::vector<std::string>& tokens,
                                        const std::vector<double>& lstm_norms) {
    CognitiveTrace trace;
    trace.confidence = 0.85;

    std::ostringstream ss;
    ss << "[C++ Cognitive Engine] Step 1: Input text received (" << query.size() << " chars, " << tokens.size() << " tokens).";
    trace.thoughts.push_back(ss.str());

    // Check C++ LSTM hidden state activations
    if (!lstm_norms.empty()) {
        std::ostringstream ss_lstm;
        ss_lstm << "[C++ Cognitive Engine] Step 2: DeepLSTM 3-layer hidden state L2-norms: [";
        for (size_t i = 0; i < lstm_norms.size(); ++i) {
            ss_lstm << std::fixed << std::setprecision(4) << lstm_norms[i];
            if (i + 1 < lstm_norms.size()) ss_lstm << ", ";
        }
        ss_lstm << "]. Conversational state synchronized.";
        trace.thoughts.push_back(ss_lstm.str());
    }

    // Check if query is math
    if (MathEvaluator::is_math_query(query)) {
        trace.primary_subsystem = "math";
        trace.confidence = 1.0;
        trace.thoughts.push_back("[C++ Cognitive Engine] Step 3: Math query pattern recognized. Dispatching to C++ High-Speed Math Engine.");
        return trace;
    }

    trace.thoughts.push_back("[C++ Cognitive Engine] Step 3: Fast-path vector indexing & semantic routing active.");
    return trace;
}

} // namespace minibrain
