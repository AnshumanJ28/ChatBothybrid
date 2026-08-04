"""Preprocessing: tokenize, lightweight spellcheck, normalize.

Features:
- Normalization (unicode NFKC, space collapse)
- Word tokenization
- Fast edit-distance fuzzy spellchecking for common domain keywords
  (e.g., 'jok' -> 'joke', 'acount' -> 'account', 'einstien' -> 'einstein')
"""
import re
import unicodedata

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Direct correction map for common abbreviations & shorthand
_COMMON_FIXES = {
    "wat": "what",
    "teh": "the",
    "plz": "please",
    "u": "you",
    "r": "are",
    "thx": "thanks",
    "helo": "hello",
    "idk": "idk",
    "nvm": "nvm",
}

# Domain vocabulary for edit-distance spellchecking
_VOCABULARY = {
    "joke", "jokes", "account", "password", "delete", "reset", "email",
    "subscription", "cancel", "einstein", "return", "order", "shipping",
    "payment", "help", "support", "billing", "login", "create", "change",
    "update", "hello", "thanks", "goodbye", "happy", "excited", "frustrated",
    "tired", "stressed", "nervous", "conscious", "neural", "network", "system"
}


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def spellcheck_token(tok: str) -> str:
    if tok in _COMMON_FIXES:
        return _COMMON_FIXES[tok]
    if tok in _VOCABULARY or len(tok) < 3 or tok.isdigit():
        return tok
    
    # Check edit distance against domain vocabulary
    best_match = tok
    min_dist = 999
    
    # Max allowed edit distance: 1 for short words (3-5), 2 for longer words (6+)
    max_allowed = 1 if len(tok) <= 5 else 2
    
    for candidate in _VOCABULARY:
        if abs(len(candidate) - len(tok)) > max_allowed:
            continue
        dist = _levenshtein(tok, candidate)
        if dist <= max_allowed and dist < min_dist:
            min_dist = dist
            best_match = candidate
            
    return best_match


def spellcheck(tokens: list[str]) -> list[str]:
    return [spellcheck_token(tok) for tok in tokens]


def preprocess(raw_text: str) -> dict:
    """Returns the cleaned text plus the token stream, ready for embedding."""
    normalized = normalize(raw_text)
    tokens = spellcheck(tokenize(normalized))
    cleaned_text = " ".join(tokens)
    return {
        "raw": raw_text,
        "normalized": normalized,
        "tokens": tokens,
        "cleaned_text": cleaned_text,
    }
