#include "math_evaluator.h"
#include <regex>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>

namespace minibrain {

// Simple recursive descent expression parser
class Parser {
    std::string expr_;
    size_t pos_;

    char peek() {
        while (pos_ < expr_.size() && std::isspace(static_cast<unsigned char>(expr_[pos_]))) {
            pos_++;
        }
        return (pos_ < expr_.size()) ? expr_[pos_] : '\0';
    }

    char get() {
        char c = peek();
        if (c != '\0') pos_++;
        return c;
    }

    double parse_number() {
        size_t start = pos_;
        while (pos_ < expr_.size() && (std::isdigit(static_cast<unsigned char>(expr_[pos_])) || expr_[pos_] == '.')) {
            pos_++;
        }
        std::string num_str = expr_.substr(start, pos_ - start);
        return std::stod(num_str);
    }

    double parse_factor() {
        char c = peek();
        if (c == '-') {
            get();
            return -parse_factor();
        }
        if (c == '+') {
            get();
            return parse_factor();
        }
        if (c == '(') {
            get(); // consume '('
            double val = parse_expression();
            if (peek() == ')') get(); // consume ')'
            return val;
        }

        if (std::isalpha(static_cast<unsigned char>(c))) {
            size_t start = pos_;
            while (pos_ < expr_.size() && std::isalpha(static_cast<unsigned char>(expr_[pos_]))) {
                pos_++;
            }
            std::string fn = expr_.substr(start, pos_ - start);
            std::transform(fn.begin(), fn.end(), fn.begin(), ::tolower);

            if (fn == "pi") return 3.141592653589793;
            if (fn == "e") return 2.718281828459045;

            // Function call e.g. sqrt(144)
            double arg = parse_factor();
            if (fn == "sqrt") return std::sqrt(arg);
            if (fn == "sin") return std::sin(arg);
            if (fn == "cos") return std::cos(arg);
            if (fn == "tan") return std::tan(arg);
            if (fn == "log" || fn == "ln") return std::log(arg);
            if (fn == "abs") return std::abs(arg);
            throw std::runtime_error("Unknown function: " + fn);
        }

        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            return parse_number();
        }

        throw std::runtime_error("Unexpected token in math expression");
    }

    double parse_power() {
        double base = parse_factor();
        while (peek() == '^') {
            get();
            double exp = parse_factor();
            base = std::pow(base, exp);
        }
        return base;
    }

    double parse_term() {
        double left = parse_power();
        while (true) {
            char op = peek();
            if (op == '*') {
                get();
                left *= parse_power();
            } else if (op == '/') {
                get();
                double right = parse_power();
                if (right == 0.0) throw std::runtime_error("Division by zero");
                left /= right;
            } else if (op == '%') {
                get();
                double right = parse_power();
                left = std::fmod(left, right);
            } else {
                break;
            }
        }
        return left;
    }

public:
    explicit Parser(const std::string& expr) : expr_(expr), pos_(0) {}

    double parse_expression() {
        double left = parse_term();
        while (true) {
            char op = peek();
            if (op == '+') {
                get();
                left += parse_term();
            } else if (op == '-') {
                get();
                left -= parse_term();
            } else {
                break;
            }
        }
        return left;
    }
};

bool MathEvaluator::is_math_query(const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Look for direct math expression patterns e.g. "35+64", "100 / 4", "sqrt(144)", "calculate 12*8"
    std::regex direct_math_re(R"(\d+\s*[+\-*/%^]\s*\d+)");
    if (std::regex_search(lower, direct_math_re)) {
        return true;
    }

    std::regex func_math_re(R"((sqrt|sin|cos|tan|log|abs)\s*\(-?\d+(\.\d+)?\))");
    return std::regex_search(lower, func_math_re);
}

std::string MathEvaluator::extract_expression(const std::string& text) {
    std::string cleaned = text;
    // Strip common question words / lead-ins
    std::regex prefix_re(R"(\b(so|what|is|are|calculate|compute|evaluate|calc|val|solve)\b)", std::regex_constants::icase);
    cleaned = std::regex_replace(cleaned, prefix_re, "");
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '?'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '!'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '='), cleaned.end());

    // Trim leading/trailing spaces
    size_t start = cleaned.find_first_not_of(" \t\n\r");
    size_t end = cleaned.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return cleaned.substr(start, end - start + 1);
}

MathResult MathEvaluator::evaluate(const std::string& query) {
    MathResult res;
    res.is_math = false;
    res.value = 0.0;

    if (!is_math_query(query)) {
        return res;
    }

    std::string expr = extract_expression(query);
    try {
        Parser parser(expr);
        double val = parser.parse_expression();

        res.is_math = true;
        res.value = val;

        std::ostringstream ss;
        // Format integer or float cleanly
        if (std::abs(val - std::round(val)) < 1e-9) {
            ss << static_cast<long long>(std::round(val));
        } else {
            ss << std::fixed << std::setprecision(4) << val;
        }

        res.formatted_result = expr + " = " + ss.str();
    } catch (const std::exception& e) {
        res.is_math = true;
        res.error = e.what();
        res.formatted_result = "Math evaluation error: " + std::string(e.what());
    }

    return res;
}

} // namespace minibrain
