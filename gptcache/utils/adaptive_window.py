"""Sliding-window load stats and load-based context window scaling (thread-safe)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Tuple

_logger = logging.getLogger(__name__)


class LoadAdaptiveMinuteWindow:
    """Track request count and token count over the last ``window_seconds`` (default 60s)."""

    __slots__ = ("_window", "_lock", "_events")

    def __init__(self, window_seconds: float = 60.0):
        self._window = float(window_seconds)
        self._lock = threading.Lock()
        self._events: Deque[Tuple[float, int, int]] = deque()

    @property
    def window_seconds(self) -> float:
        return self._window

    def record_request(self, input_tokens: int) -> None:
        now = time.time()
        tok = max(0, int(input_tokens))
        with self._lock:
            self._events.append((now, 1, tok))
            self._prune_locked(now)

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def counts_in_window(self) -> Tuple[int, int]:
        """Return ``(request_count, total_input_tokens)`` in the current window."""
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            req_total = 0
            tok_total = 0
            for _, r, t in self._events:
                req_total += r
                tok_total += t
            return req_total, tok_total


class LoadAdaptiveContextController:
    """Minute load counters + compare adjacent minutes (R× / 1/R×) to adjust ``context_cache_overall_factor``."""

    __slots__ = ("_cache", "_window", "_ctrl_lock", "_eval_anchor", "_prev_req", "_prev_tok")

    def __init__(self, cache: Any, window_seconds: float = 60.0):
        self._cache = cache
        self._window = LoadAdaptiveMinuteWindow(window_seconds)
        self._ctrl_lock = threading.Lock()
        self._eval_anchor = time.time()
        self._prev_req = 0
        self._prev_tok = 0

    def record_request(self, input_tokens: int) -> None:
        self._window.record_request(input_tokens)
        with self._ctrl_lock:
            now = time.time()
            if now - self._eval_anchor < self._window.window_seconds:
                return
            curr_req, curr_tok = self._window.counts_in_window()
            prev_req, prev_tok = self._prev_req, self._prev_tok
            self._maybe_resize_context_window(prev_req, prev_tok, curr_req, curr_tok)
            self._prev_req, self._prev_tok = curr_req, curr_tok
            self._eval_anchor = now

    def minute_counts(self) -> Tuple[int, int]:
        return self._window.counts_in_window()

    def _maybe_resize_context_window(
        self,
        prev_req: int,
        prev_tok: int,
        curr_req: int,
        curr_tok: int,
    ) -> None:
        cfg = self._cache.config
        n = int(cfg.context_cache_window_len)
        if n < 1:
            return
        w_min = int(getattr(cfg, "context_cache_window_min", 2))
        w_max = int(cfg.context_cache_window_max)
        w = cfg.effective_context_window_len(None)
        ratio_req = float(cfg.load_adaptive_ratio)
        ratio_tok = float(getattr(cfg, "load_adaptive_token_ratio", ratio_req))
        if ratio_tok <= 1.0:
            ratio_tok = ratio_req

        high_load = False
        low_load = False
        if prev_req > 0:
            if curr_req >= ratio_req * prev_req:
                high_load = True
            elif curr_req <= prev_req / ratio_req:
                low_load = True
        if prev_tok > 0:
            if curr_tok >= ratio_tok * prev_tok:
                high_load = True
            elif curr_tok <= prev_tok / ratio_tok:
                low_load = True

        window_s = self._window.window_seconds
        curr_rps = (curr_req / window_s) if window_s > 0 else 0.0
        shrink_min_rps = float(getattr(cfg, "load_adaptive_shrink_min_rps", 0.0))
        grow_max_rps = float(getattr(cfg, "load_adaptive_grow_max_rps", 0.0))
        force_shrink_rps = float(getattr(cfg, "load_adaptive_force_shrink_rps", 0.0))
        bypass_min_req = float(
            getattr(cfg, "load_adaptive_shrink_spike_bypass_min_prev_req", 0.0)
        )
        bypass_min_tok = float(
            getattr(cfg, "load_adaptive_shrink_spike_bypass_min_prev_tok", 0.0)
        )

        if high_load and shrink_min_rps > 0.0 and curr_rps < shrink_min_rps:
            strong_req_spike = (
                bypass_min_req > 0.0
                and prev_req >= bypass_min_req
                and curr_req >= ratio_req * prev_req
            )
            strong_tok_spike = (
                bypass_min_tok > 0.0
                and prev_tok >= bypass_min_tok
                and curr_tok >= ratio_tok * prev_tok
            )
            if strong_req_spike or strong_tok_spike:
                _logger.info(
                    "load_adaptive: shrink_min bypass (strong spike; curr_rps=%.3f < shrink_min=%.3f); "
                    "req %s→%s, tok %s→%s; keeping high_load (req_R=%.3g tok_R=%.3g)",
                    curr_rps,
                    shrink_min_rps,
                    prev_req,
                    curr_req,
                    prev_tok,
                    curr_tok,
                    ratio_req,
                    ratio_tok,
                )
            else:
                _logger.info(
                    "load_adaptive: ratio spike suppressed (curr_rps=%.3f < shrink_min=%.3f); "
                    "req %s→%s, tok %s→%s; effective window stays %s",
                    curr_rps,
                    shrink_min_rps,
                    prev_req,
                    curr_req,
                    prev_tok,
                    curr_tok,
                    w,
                )
                high_load = False
        if low_load and grow_max_rps > 0.0 and curr_rps > grow_max_rps:
            _logger.info(
                "load_adaptive: ratio drop suppressed (curr_rps=%.3f > grow_max=%.3f); "
                "req %s→%s, tok %s→%s; effective window stays %s",
                curr_rps,
                grow_max_rps,
                prev_req,
                curr_req,
                prev_tok,
                curr_tok,
                w,
            )
            low_load = False

        if (
            force_shrink_rps > 0.0
            and curr_rps >= force_shrink_rps
            and not low_load
        ):
            if not high_load:
                _logger.info(
                    "load_adaptive: force-shrink (curr_rps=%.3f >= force=%.3f); "
                    "req %s→%s, tok %s→%s; effective window %s",
                    curr_rps,
                    force_shrink_rps,
                    prev_req,
                    curr_req,
                    prev_tok,
                    curr_tok,
                    w,
                )
            high_load = True

        if high_load and low_load:
            return
        if high_load:
            new_w = max(w_min, w - 1)
            if new_w != w:
                cfg.context_cache_overall_factor = new_w / float(n)
                _logger.info(
                    "load_adaptive: high load vs previous minute (req_R=%.3g tok_R=%.3g; "
                    "req %s→%s, tok %s→%s); effective window %s → %s (overall_factor=%.6g)",
                    ratio_req,
                    ratio_tok,
                    prev_req,
                    curr_req,
                    prev_tok,
                    curr_tok,
                    w,
                    new_w,
                    cfg.context_cache_overall_factor,
                )
            return
        if low_load:
            new_w = min(w_max, w + 1)
            if new_w != w:
                cfg.context_cache_overall_factor = new_w / float(n)
                _logger.info(
                    "load_adaptive: low load vs previous minute (req_R=%.3g tok_R=%.3g; "
                    "req %s→%s, tok %s→%s); effective window %s → %s (overall_factor=%.6g)",
                    ratio_req,
                    ratio_tok,
                    prev_req,
                    curr_req,
                    prev_tok,
                    curr_tok,
                    w,
                    new_w,
                    cfg.context_cache_overall_factor,
                )
