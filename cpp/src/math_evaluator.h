#pragma once
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <stdexcept>
#include <cctype>

namespace minibrain {

struct MathResult {
    bool is_math;
    double value;
    std::string formatted_result;
    std::string error;
};

class MathEvaluator {
public:
    // Detects if a text string is a math query (e.g., "what is 35+64", "calc 12 * 8", "sqrt(144)")
    static bool is_math_query(const std::string& text);

    // Parses and evaluates a mathematical expression string in C++
    static MathResult evaluate(const std::string& query);

private:
    static std::string extract_expression(const std::string& text);
    static double parse_expression(const std::string& expr);
};

} // namespace minibrain
