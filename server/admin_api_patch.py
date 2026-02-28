"""
MLX Server Admin API Patch
===========================
注入 admin 端点，支持按需 unload/load 模型。

端点：
    GET  /v1/admin/models                    — 列出所有模型及状态
    GET  /v1/admin/models/{model_id}/stats   — 查询队列状态
    POST /v1/admin/models/{model_id}/unload  — 卸载模型释放内存
    POST /v1/admin/models/{model_id}/load    — 重新加载模型
"""

from __future__ import annotations

import dataclasses
import time
from http import HTTPStatus
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.config import ModelEntryConfig
from app.core.handler_process import HandlerProcessProxy
from lazy_handler_proxy import LazyHandlerProxy

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

admin_router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _registry(request: Request):
    r = getattr(request.app.state, "registry", None)
    if r is None:
        raise HTTPException(503, "Registry not available (single-model mode)")
    return r


def _load_model_cfg(model_id: str) -> ModelEntryConfig:
    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    entry = next((m for m in raw.get("models", []) if m.get("model_id") == model_id), None)
    if entry is None:
        raise HTTPException(404, f"Model '{model_id}' not found in config.yaml")
    return ModelEntryConfig(**entry)


def _lazy_flags_for(model_id: str) -> tuple[bool, int]:
    """Return (is_lazy, idle_timeout) for a model from config.yaml."""
    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    entry = next((m for m in raw.get("models", []) if m.get("model_id") == model_id), None)
    if entry is None:
        return False, 0
    return entry.get("lazy", False), entry.get("idle_timeout", 1800)


@admin_router.get("/models")
async def admin_list_models(request: Request):
    return {"models": _registry(request).list_models()}


@admin_router.get("/models/{model_id}/stats")
async def admin_model_stats(model_id: str, request: Request):
    reg = _registry(request)
    if not reg.has_model(model_id):
        raise HTTPException(404, f"Model '{model_id}' not loaded")
    stats = await reg.get_handler(model_id).get_queue_stats()
    return {"model_id": model_id, "queue_stats": stats}


@admin_router.post("/models/{model_id}/unload")
async def admin_unload_model(model_id: str, request: Request):
    reg = _registry(request)
    if not reg.has_model(model_id):
        raise HTTPException(404, f"Model '{model_id}' not loaded")

    # 拒绝卸载有活跃请求的模型
    try:
        stats = await reg.get_handler(model_id).get_queue_stats()
        active = stats.get("queue_stats", stats).get("active_requests", 0)
        if active > 0:
            raise HTTPException(409, f"Model '{model_id}' has {active} active request(s)")
    except HTTPException:
        raise
    except Exception:
        pass

    logger.info(f"Admin: unloading '{model_id}'")
    await reg.unregister_model(model_id)

    # Fix 4 & 6: sync LazyHandlerProxy state and clean lazy_proxies dict
    lazy_proxies = getattr(request.app.state, "lazy_proxies", {})
    lp_entry = lazy_proxies.get(model_id)
    if lp_entry is not None:
        lp_entry[0]._started = False
        del lazy_proxies[model_id]

    logger.info(f"Admin: '{model_id}' unloaded ✓")
    return {"status": "unloaded", "model_id": model_id, "timestamp": int(time.time())}


@admin_router.post("/models/{model_id}/load")
async def admin_load_model(model_id: str, request: Request):
    reg = _registry(request)
    if reg.has_model(model_id):
        return {"status": "already_loaded", "model_id": model_id}

    cfg = _load_model_cfg(model_id)
    is_lazy, idle_timeout = _lazy_flags_for(model_id)
    logger.info(f"Admin: loading '{model_id}' from {cfg.model_path}")

    proxy = HandlerProcessProxy(
        model_cfg_dict=dataclasses.asdict(cfg),
        model_type=cfg.model_type,
        model_path=cfg.model_path,
        model_id=model_id,
    )
    queue_config = {
        "max_concurrency": cfg.max_concurrency,
        "timeout": cfg.queue_timeout,
        "queue_size": cfg.queue_size,
    }

    # Fix 5 & 6: restore lazy wrapper if model is configured as lazy
    if is_lazy:
        handler = LazyHandlerProxy(proxy, queue_config)
        await handler._ensure_started()
        lazy_proxies = getattr(request.app.state, "lazy_proxies", {})
        lazy_proxies[model_id] = (handler, idle_timeout)
        request.app.state.lazy_proxies = lazy_proxies
    else:
        handler = proxy
        await proxy.start(queue_config)

    await reg.register_model(
        model_id=model_id,
        handler=handler,
        model_type=cfg.model_type,
        context_length=cfg.context_length,
    )

    logger.info(f"Admin: '{model_id}' loaded ✓")
    return {"status": "loaded", "model_id": model_id, "model_path": cfg.model_path}


def install(app):
    app.include_router(admin_router)
    logger.info("Admin API installed: /v1/admin/models/*")
