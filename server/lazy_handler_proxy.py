"""
Lazy-loading wrapper for HandlerProcessProxy.
Model subprocess is only spawned on first request, not at server startup.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.handler_process import HandlerProcessProxy


class LazyHandlerProxy:
    """Wraps HandlerProcessProxy — defers start() until first request."""

    def __init__(self, proxy: HandlerProcessProxy, queue_config: dict):
        self._proxy = proxy
        self._queue_config = queue_config
        self._started = False
        self._lock = asyncio.Lock()
        self.last_request_time: float = 0.0
        # Save factory params so we can rebuild the proxy after cleanup
        self._proxy_factory = {
            "model_cfg_dict": proxy._model_cfg_dict,
            "model_type": proxy._model_cfg_dict.get("model_type", "lm"),
            "model_path": proxy.model_path,
            "model_id": proxy.model_id,
        }

    async def _ensure_started(self):
        if self._started:
            return
        async with self._lock:
            if not self._started:
                await self._proxy.start(self._queue_config)
                self._started = True

    def is_loaded(self) -> bool:
        return self._started

    async def unload(self):
        async with self._lock:
            if not self._started:
                return
            await self._proxy.cleanup()
            self._started = False
            # HandlerProcessProxy is single-use after cleanup; rebuild it
            self._proxy = HandlerProcessProxy(**self._proxy_factory)

    # ── delegate all handler methods ──────────────────────────────────────

    def __getattr__(self, name: str):
        attr = getattr(self._proxy, name)
        if not asyncio.iscoroutinefunction(attr):
            return attr

        async def wrapper(*args, **kwargs):
            await self._ensure_started()
            self.last_request_time = time.monotonic()
            return await attr(*args, **kwargs)

        return wrapper

    # async generators need special handling
    async def generate_text_stream(self, *a, **kw):
        await self._ensure_started()
        self.last_request_time = time.monotonic()
        async for chunk in self._proxy.generate_text_stream(*a, **kw):
            yield chunk

    async def generate_multimodal_stream(self, *a, **kw):
        await self._ensure_started()
        self.last_request_time = time.monotonic()
        async for chunk in self._proxy.generate_multimodal_stream(*a, **kw):
            yield chunk

    async def generate_transcription_stream_from_data(self, *a, **kw):
        await self._ensure_started()
        self.last_request_time = time.monotonic()
        async for chunk in self._proxy.generate_transcription_stream_from_data(*a, **kw):
            yield chunk
