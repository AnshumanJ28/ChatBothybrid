"""Simple in-memory TTL cache for tool-dispatch results (e.g. web search).

Keeps latency and API quota usage sane for repeated/near-duplicate
queries within a session or short window. Swap for Redis etc. in
production without touching callers — the interface is get/set only.
"""
import time


class TTLCache:
    def __init__(self, ttl_seconds: int = 600):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}

    def _normalize_key(self, key: str) -> str:
        return " ".join(key.lower().split())

    def get(self, key: str):
        k = self._normalize_key(key)
        entry = self._store.get(k)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            del self._store[k]
            return None
        return value

    def set(self, key: str, value):
        k = self._normalize_key(key)
        self._store[k] = (time.time() + self.ttl_seconds, value)

    def clear(self):
        self._store.clear()
