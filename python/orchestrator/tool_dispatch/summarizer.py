"""
Classical NLP Extractive Summarizer.
No LLM used by default. Combines query-relevance overlaps and sentence centrality (cosine similarity to centroid)
to select the most representative, non-redundant sentences from web content or snippets.
If GEMINI_API_KEY is present, automatically upgrades to generative web search summarization.
"""
import os
import json
import urllib.request
import re
import math
import collections
from .. import embedding as _embedding
from .. import minibrain_cpp as _mb

_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:generateContent"

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def split_sentences(text: str) -> list[str]:
    """Segment raw text into sentences while respecting common abbreviations."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split using punctuation followed by spaces
    parts = re.split(r'(\. |\? |\! )', text)
    sentences = []
    current = ""
    abbreviations = {"mr.", "dr.", "ms.", "mrs.", "co.", "inc.", "e.g.", "i.e.", "vs.", "etc.", "st.", "[stub]"}
    
    for part in parts:
        if part is None:
            continue
        current += part
        if part in {". ", "? ", "! "}:
            words = current.strip().lower().split()
            # If the last word is in our abbreviations list, keep building this sentence
            if words and words[-1] in abbreviations:
                continue
            sentences.append(current.strip())
            current = ""
            
    if current.strip():
        sentences.append(current.strip())
        
    # Return sentences that have at least 3 words and look like real text
    return [s for s in sentences if len(s.split()) >= 3]

def clean_and_tokenize(text: str) -> list[str]:
    """Lowercase text, remove punctuation, and strip stopwords."""
    cleaned = re.sub(r'[^a-z0-9\s]', '', text.lower())
    tokens = cleaned.split()
    return [w for w in tokens if w not in STOPWORDS]

NOISE_INDICATORS = [
    'vte', 'category:', 'categories:', 'authority control', 'copley medallists',
    'nobel prize recipients', 'fellows of the royal society', 'retrieved from',
    'works by', 'articles from', 'isbn', 'issn', 'doi', 's2cid', 'digital commons',
    'project gutenberg', 'internet archive', 'librivox', 'external links', 'further reading',
    'see also', 'main article', 'edit links', 'wikisource', 'bibcode', 'pmc', 'pmid',
    'wikipedia', 'wikiquote', 'wikidata', 'vtealbert', 'vtecopley', 'archived from the original',
    'jstor', 'oclc', 'hdl', 'pp.', 'p. ', 'vol.', 'journal of', 'proceedings of'
]

# Regex patterns typical of bibliography/citation entries, not prose sentences.
_CITATION_PATTERNS = [
    re.compile(r'^\s*[↑\^]'),                       # leading footnote-back arrow
    re.compile(r'"\s*\.\s*$'),                       # ends with a quoted title + period
    re.compile(r'\(\d{4}[a-z]?\)'),                  # (1917a)-style citation year
    re.compile(r'^\s*\w[\w.\-]*,\s*\w[\w.\-]*\s*\('), # "Lastname, F. (" citation-author lead-in
]


def is_clean_sentence(sentence: str) -> bool:
    s_lower = sentence.lower()
    if any(ind in s_lower for ind in NOISE_INDICATORS):
        return False
    if any(pat.search(sentence) for pat in _CITATION_PATTERNS):
        return False
    words = sentence.split()
    if len(words) < 3 or len(words) > 55:
        return False
    num_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    if num_caps / max(len(words), 1) > 0.35:
        return False
    # Citation lines are quote-heavy relative to their length
    if sentence.count('"') >= 2 and len(words) < 20:
        return False
    return True


def summarize_text(text: str, query: str, num_sentences: int = 3) -> dict:
    """Extracts top `num_sentences` relevant to `query` from `text`.
    Returns: { 'summary': str, 'logs': list[str] }
    """
    if _API_KEY:
        logs = ["Using live Gemini API for generative web search summarization..."]
        url = f"{_ENDPOINT}?key={_API_KEY}"
        
        prompt = (
            f"You are a web search summarizer. Synthesize a professional, concise summary "
            f"(about {num_sentences} sentences long) of the following crawled web text based on the "
            f"user query: '{query}'. Provide ONLY the summary itself, nothing else.\n\n"
            f"Crawled Text:\n{text[:6000]}"
        )
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 200,
                "temperature": 0.3
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                res_data = json.loads(response.read().decode("utf-8"))
            
            candidates = res_data.get("candidates", [])
            if candidates:
                part = candidates[0].get("content", {}).get("parts", [{}])[0]
                summary = part.get("text", "").strip()
                if summary:
                    logs.append("Generative web synthesis complete.")
                    return {"summary": summary, "logs": logs}
            logs.append("Generative synthesis response empty or invalid. Falling back to classical NLP summarizer.")
        except Exception as e:
            logs.append(f"Generative synthesis request failed: {e}. Falling back to classical NLP summarizer.")

    raw_sentences = split_sentences(text)
    sentences = [s for s in raw_sentences if is_clean_sentence(s)]
    if not sentences:
        sentences = [s for s in raw_sentences if len(s.split()) >= 4][:5]
    if not sentences:
        return {"summary": "", "logs": ["No sentences found to summarize."]}
    
    if len(sentences) <= num_sentences:
        return {
            "summary": " ".join(sentences),
            "logs": ["Source text is already concise; returning all sentences."]
        }
    
    query_tokens = set(clean_and_tokenize(query))
    logs = [f"Analyzing {len(sentences)} clean sentences (filtered from {len(raw_sentences)} raw)..."]
    
    # 1. Compute term frequencies over the entire text
    all_words = []
    sentence_tokens = []
    for s in sentences:
        toks = clean_and_tokenize(s)
        sentence_tokens.append(toks)
        all_words.extend(toks)
        
    word_counts = collections.Counter(all_words)
    total_words = sum(word_counts.values()) or 1
    tf = {w: count / total_words for w, count in word_counts.items()}
    
    # 2. Score sentences via C++ semantic similarity (AttentionPooling + EmbeddingIndex),
    # same cosine-similarity engine KB Search uses, instead of raw keyword-overlap counting.
    try:
        sent_vectors = [_embedding.encode(s) for s in sentences]
        doc_centroid = _mb.AttentionPooling.pool(sent_vectors)

        sent_index = _mb.EmbeddingIndex(_embedding.EMBED_DIM)
        for i, vec in enumerate(sent_vectors):
            sent_index.add(vec, str(i))

        # Query relevance (falls back to doc centroid when query has no content words)
        query_vec = _embedding.encode(query) if query_tokens else doc_centroid
        ranked = sent_index.search(query_vec, len(sentences))

        # Centrality: how similar each sentence is to the whole document's meaning
        centrality_index = _mb.EmbeddingIndex(_embedding.EMBED_DIM)
        for i, vec in enumerate(sent_vectors):
            centrality_index.add(vec, str(i))
        centrality_by_idx = {int(r.index): r.score for r in centrality_index.search(doc_centroid, len(sentences))}

        query_score_by_idx = {int(r.index): r.score for r in ranked}
        scored_sentences = []
        for idx, (s, toks) in enumerate(zip(sentences, sentence_tokens)):
            if not toks:
                continue
            query_score = query_score_by_idx.get(idx, 0.0)
            centrality = centrality_by_idx.get(idx, 0.0)
            final_score = (0.7 * query_score + 0.3 * centrality) if query_tokens else centrality
            scored_sentences.append({
                "index": idx,
                "text": s,
                "tokens": toks,
                "query_score": query_score,
                "centrality": centrality,
                "score": final_score
            })
        logs.append("Scored sentences using C++ EmbeddingIndex cosine similarity (semantic, not keyword count).")
    except Exception as e:
        logs.append(f"C++ embedding scoring unavailable ({e}); falling back to TF overlap scoring.")
        doc_norm = math.sqrt(sum(count * count for count in word_counts.values())) or 1e-9
        scored_sentences = []
        for idx, (s, toks) in enumerate(zip(sentences, sentence_tokens)):
            if not toks:
                continue
            query_score = sum(tf.get(tok, 0.0) * 100.0 for tok in toks if tok in query_tokens)
            sent_vector = collections.Counter(toks)
            sent_norm = math.sqrt(sum(count * count for count in sent_vector.values())) or 1e-9
            dot_product = sum(sent_vector[w] * word_counts[w] for w in sent_vector)
            centrality = dot_product / (sent_norm * doc_norm)
            final_score = (0.7 * query_score + 0.3 * centrality) if query_tokens else centrality
            scored_sentences.append({
                "index": idx, "text": s, "tokens": toks,
                "query_score": query_score, "centrality": centrality, "score": final_score
            })

    scored_sentences.sort(key=lambda x: x["score"], reverse=True)
    
    # 3. Prevent redundancy using Jaccard Similarity check
    selected = []
    logs.append("Ranking sentences by relevance and centrality...")
    for candidate in scored_sentences:
        if len(selected) >= num_sentences:
            break
            
        cand_tokens = set(candidate["tokens"])
        is_redundant = False
        for sel in selected:
            sel_tokens = set(sel["tokens"])
            union = cand_tokens.union(sel_tokens)
            if union:
                jaccard = len(cand_tokens.intersection(sel_tokens)) / len(union)
                if jaccard > 0.40:
                    is_redundant = True
                    logs.append(f"Skipped redundant sentence (Jaccard similarity {jaccard:.2f}): '{candidate['text'][:50]}...'")
                    break
                    
        if not is_redundant:
            selected.append(candidate)
            logs.append(f"Ranked sentence selected: '{candidate['text'][:60]}...' (Score: {candidate['score']:.4f})")
            
    # Restore original document order
    selected.sort(key=lambda x: x["index"])
    
    summary = " ".join([s["text"] for s in selected])
    return {
        "summary": summary,
        "logs": logs
    }