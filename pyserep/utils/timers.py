"""pyserep.utils.timers — simple timing utilities."""

from __future__ import annotations

import time


class Timer:
    """Context-manager wall-clock timer."""

    def __init__(self, label: str = "", verbose: bool = True) -> None:
        self.label   = label
        self.verbose = verbose
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._t0
        if self.verbose and self.label:
            print(f"  [{self.label}] {self.elapsed:.3f}s")
