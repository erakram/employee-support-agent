"""A small in-memory cache whose entries expire after a fixed time (TTL)."""

import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: float, max_size: int = 256):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None or entry[0] < time.monotonic():
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)  # mark as recently used
        self.hits += 1
        return entry[1]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)  # evict the least recently used entry
