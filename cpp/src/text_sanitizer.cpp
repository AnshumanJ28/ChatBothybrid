#include "text_sanitizer.h"
#include <regex>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace minibrain {

std::string TextSanitizer::remove_citations(const std::string& text) {
    // Regex matching [ 3 ], [40], [ 41 ], [stub], etc.
    std::regex citation_re(R"(\[\s*\d+\s*\]|\[\s*stub\s*\])");
    return std::regex_replace(text, citation_re, "");
}

std::string TextSanitizer::remove_unit_tables(const std::string& text) {
    // If text starts with raw unit table listings like "SI base units 1 kg ⋅ m..."
    // strip everything up to the first actual sentence like "The newton (symbol: N)..."
    std::regex unit_prefix_re(R"(^(SI base units|CGS units|Imperial units)[^A-Z]*?(?=[A-Z][a-z]+ (is|was|are|were|refers|denotes|defined)))");
    std::string result = std::regex_replace(text, unit_prefix_re, "");
    return result;
}

std::string TextSanitizer::remove_nav_junk(const std::string& text) {
    std::string result = text;

    // Remove leading Wikipedia section titles like "[ 40 ] [ 41 ] Marriages, relationships and children Albert Einstein and..."
    std::regex section_header_re(R"(^(\s*\[\s*\d+\s*\])*\s*(Marriages|Early life|Career|Biography|Legacy|References|External links|See also)[^A-Z]*?(?=[A-Z][a-z]+))");
    result = std::regex_replace(result, section_header_re, "");

    // Remove 'vte' navigation lists, Copley Medallists, etc.
    std::regex vte_re(R"(vte[A-Za-z0-9\s_\-\(\)]+)");
    result = std::regex_replace(result, vte_re, "");

    // Collapse multiple spaces
    std::regex spaces_re(R"(\s+)");
    result = std::regex_replace(result, spaces_re, " ");

    // Trim leading/trailing spaces
    size_t start = result.find_first_not_of(" \t\n\r");
    size_t end = result.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return result.substr(start, end - start + 1);
}

std::string TextSanitizer::sanitize(const std::string& raw_text) {
    std::string step1 = remove_citations(raw_text);
    std::string step2 = remove_unit_tables(step1);
    std::string step3 = remove_nav_junk(step2);

    // Final clean check: ensure text starts at first capital letter if preceded by stray numbers or symbols
    size_t cap_pos = 0;
    while (cap_pos < step3.size() && !std::isupper(static_cast<unsigned char>(step3[cap_pos]))) {
        cap_pos++;
    }
    if (cap_pos > 0 && cap_pos < step3.size() && (cap_pos <= 15)) {
        step3 = step3.substr(cap_pos);
    }
    return step3;
}

} // namespace minibrain
