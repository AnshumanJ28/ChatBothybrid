"""Web search tool dispatch: retrieval against the live web, same pattern
as KB Search but for the overflow case. NOT generation — the reply is
either the top snippet dropped into a template, or a disambiguation
list, exactly like KB Search's Composer pattern.

Uses live DuckDuckGo HTML search (no API key required). Falls back to
a stub responder (clearly marked) when DuckDuckGo can't be reached,
so the rest of the pipeline is runnable/testable without network
access.
"""
import urllib.request
import urllib.parse
import re
from html.parser import HTMLParser
from .cache import TTLCache
from . import summarizer

_cache = TTLCache(ttl_seconds=600)


class WebSearchError(Exception):
    pass


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.ignore_depth = 0
        self.ignored_tags = {'script', 'style', 'head', 'title', 'meta', 'link', 'noscript', 'header', 'footer', 'nav', 'table', 'aside', 'form', 'button', 'sup'}
        self.ignored_keywords = {
            'sidebar', 'nav', 'footer', 'catlinks', 'mw-navigation', 'vector-menu', 'vte',
            'authority-control', 'infobox', 'toc', 'navigation', 'mw-data-after-content',
            'reflist', 'reference', 'cite_note', 'citation', 'bibliography', 'refbegin',
            'refend', 'shortdescription', 'hatnote', 'printfooter', 'mw-editsection',
            'ambox', 'navbox', 'external-links', 'further-reading'
        }

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls_and_id = (attrs_dict.get("class", "") + " " + attrs_dict.get("id", "")).lower()
        if tag in self.ignored_tags or any(kw in cls_and_id for kw in self.ignored_keywords):
            self.ignore_depth += 1

    def handle_endtag(self, tag):
        if self.ignore_depth > 0:
            self.ignore_depth -= 1

    def handle_data(self, data):
        if self.ignore_depth == 0:
            self.text_parts.append(data)

    def get_text(self) -> str:
        raw_text = " ".join(self.text_parts)
        return re.sub(r'\s+', ' ', raw_text).strip()


def scrape_url(url: str, timeout: float = 2.5) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        html = response.read().decode('utf-8', errors='ignore')
    
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


class DDGParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_result = {}
        self.in_result = False
        self.in_title_link = False
        self.in_snippet = False
        self.title_text = []
        self.snippet_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")
        if tag == "div" and "result" in cls.split():
            self.in_result = True
            self.current_result = {}
        elif self.in_result and tag == "a" and "result__a" in cls.split():
            self.in_title_link = True
            self.current_result["url"] = attrs_dict.get("href", "")
            self.title_text = []
        elif self.in_result and tag == "a" and "result__snippet" in cls.split():
            self.in_snippet = True
            self.snippet_text = []

    def handle_endtag(self, tag):
        if self.in_title_link and tag == "a":
            self.in_title_link = False
            self.current_result["title"] = "".join(self.title_text).strip()
        elif self.in_snippet and tag == "a":
            self.in_snippet = False
            self.current_result["snippet"] = "".join(self.snippet_text).strip()
            if "url" in self.current_result and self.current_result.get("url"):
                url = self.current_result["url"]
                if "uddg=" in url:
                    match = re.search(r'uddg=([^&]+)', url)
                    if match:
                        url = urllib.parse.unquote(match.group(1))
                self.current_result["url"] = url
                self.results.append(self.current_result)
            self.in_result = False

    def handle_data(self, data):
        if self.in_title_link:
            self.title_text.append(data)
        elif self.in_snippet:
            self.snippet_text.append(data)


def _ddg_search(query: str, num: int = 3) -> list[dict]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
    
    parser = DDGParser()
    parser.feed(html)
    return parser.results[:num]


def _stub_search(query: str, num: int = 3) -> list[dict]:
    """Deterministic offline stand-in used when no API credentials are
    configured, so the router/composer path is exercisable in tests."""
    return [
        {
            "title": f"[stub] No live web search available for: {query}",
            "snippet": "DuckDuckGo search could not be reached. Python has built-in offline search logic that can process text snippets.",
            "url": "https://example.com/mock-search-result-1"
        },
        {
            "title": "Offline NLP Chatbot Architecture",
            "snippet": "This system features a hybrid classical NLP pipeline, utilizing a C++ compiled index for embedding-based search, a custom flow engine, and an offline web summarizer.",
            "url": "https://example.com/mock-search-result-2"
        }
    ]


def extract_url(text: str) -> str | None:
    match = re.search(r'(https?://[^\s]+)', text)
    return match.group(1) if match else None


def search(query: str, num: int = 3) -> dict:
    """Returns {'results': [...], 'summary': str, 'sources': [...], 'logs': [...], 'from_cache': bool, 'error': str|None}."""
    logs = []
    
    # Check if query itself is or contains a URL
    url_target = extract_url(query)
    if url_target:
        logs.append(f"Detected direct URL target: {url_target}")
        try:
            logs.append(f"Scraping text content from {url_target}...")
            scraped_text = scrape_url(url_target)
            logs.append(f"Successfully scraped {len(scraped_text)} characters. Running extractive summarizer...")
            summary_res = summarizer.summarize_text(scraped_text, query, num_sentences=4)
            logs.extend(summary_res["logs"])
            
            return {
                "results": [{"title": "Direct URL Scrape", "snippet": summary_res["summary"], "url": url_target}],
                "summary": summary_res["summary"],
                "sources": [url_target],
                "logs": logs,
                "from_cache": False,
                "error": None
            }
        except Exception as e:
            logs.append(f"Failed to scrape URL direct target: {e}. Falling back to standard search.")

    cached = _cache.get(query)
    if cached is not None:
        return {
            "results": cached["results"],
            "summary": cached["summary"],
            "sources": cached["sources"],
            "logs": ["Loaded web search results and summary from TTL Cache."],
            "from_cache": True,
            "error": None
        }

    results = []
    error_msg = None
    logs.append(f"Invoking DuckDuckGo Search for: '{query}'")
    try:
        results = _ddg_search(query, num)
    except Exception as ddg_err:
        logs.append(f"DuckDuckGo search failed: {ddg_err}. Falling back to offline stub.")
        results = _stub_search(query, num)

    if not results:
        return {
            "results": [],
            "summary": "I couldn't find any search results to summarize.",
            "sources": [],
            "logs": logs,
            "from_cache": False,
            "error": error_msg or "zero_results"
        }

    # Now attempt to crawl top 1-2 URLs and summarize
    sources = [r["url"] for r in results if r["url"]]
    scraped_any = False
    combined_texts = []
    
    for idx, r in enumerate(results[:2]):
        url = r["url"]
        if not url or url.startswith("https://example.com"):
            continue
        try:
            logs.append(f"Crawl target [{idx+1}]: {url}")
            page_text = scrape_url(url, timeout=2.0)
            if len(page_text.strip()) > 100:
                logs.append(f"Successfully scraped {len(page_text)} chars from {url}")
                combined_texts.append(page_text)
                scraped_any = True
            else:
                logs.append(f"Scraped content from {url} is too short.")
        except Exception as e:
            logs.append(f"Could not scrape {url}: {e}")

    # If we couldn't scrape any actual live web pages, we summarize the search snippets!
    if not scraped_any:
        logs.append("No live pages scraped. Combining search snippets for summarization instead.")
        snippet_text = " ".join([f"{r['title']}. {r['snippet']}" for r in results])
        combined_texts.append(snippet_text)

    full_corpus = "\n\n".join(combined_texts)
    try:
        from .. import minibrain_cpp as mb
        full_corpus = mb.TextSanitizer.sanitize(full_corpus)
    except Exception:
        pass
    
    logs.append("Running extractive summarization on compiled text...")
    summary_res = summarizer.summarize_text(full_corpus, query, num_sentences=3)
    logs.extend(summary_res["logs"])
    
    final_data = {
        "results": results,
        "summary": summary_res["summary"],
        "sources": sources,
        "logs": logs,
        "from_cache": False,
        "error": None
    }
    
    _cache.set(query, final_data)
    return final_data