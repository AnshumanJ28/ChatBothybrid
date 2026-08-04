#pragma once
#include <string>
#include <vector>

namespace minibrain {

class TextSanitizer {
public:
    // Strips citation brackets like [ 3 ], [40], [ 41 ], etc.
    static std::string remove_citations(const std::string& text);

    // Removes raw unit conversion junk (e.g. SI base units 1 kg ⋅ m ⋅ s −2 CGS units...)
    static std::string remove_unit_tables(const std::string& text);

    // Removes section headers, navigation artifacts, vte templates
    static std::string remove_nav_junk(const std::string& text);

    // Full sanitization pipeline for clean, crisp output
    static std::string sanitize(const std::string& raw_text);
};

} // namespace minibrain
