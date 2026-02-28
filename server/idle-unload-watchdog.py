#!/usr/bin/env python3
"""
MLX Server Idle Unload Watchdog
================================
监控 mlx-openai-server 上各模型的最后请求时间，
超过 idle_timeout 后自动 unload，有新请求时自动 reload。

用法:
    python idle-unload-watchdog.py [--config config.yaml] [--watchdog-config watchdog.yaml]

原理:
    1. 每隔 check_interval 秒轮询 /v1/models/{id}/last_used（通过 middleware 注入）
    2. 超时则 POST /v1/admin/models/{id}/unload
    3. 收到 404 时自动 POST /v1/admin/models/{id}/load 重新加载
"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("idle-watchdog")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelWatchConfig:
    model_id: str
    idle_timeout: int          # seconds, 0 = never unload
    always_loaded: bool = False


@dataclass
class WatchdogConfig:
    base_url: str = "http://127.0.0.1:8787"
    check_interval: int = 60   # seconds between checks
    admin_token: str = ""      # optional Bearer token for admin endpoints
    models: list[ModelWatchConfig] = field(default_factory=list)


def load_watchdog_config(path: str) -> WatchdogConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Watchdog config not found: {path}")
    raw = yaml.safe_load(p.read_text())
    cfg = WatchdogConfig(
        base_url=raw.get("base_url", "http://127.0.0.1:8787").rstrip("/"),
        check_interval=raw.get("check_interval", 60),
        admin_token=raw.get("admin_token", ""),
    )
    for m in raw.get("models", []):
        cfg.models.append(ModelWatchConfig(
            model_id=m["model_id"],
            idle_timeout=m.get("idle_timeout", 0),
            always_loaded=m.get("always_loaded", False),
        ))
    return cfg


# ---------------------------------------------------------------------------
# Watchdog state
# ---------------------------------------------------------------------------

class ModelState:
    def __init__(self, model_id: str, idle_timeout: int, always_loaded: bool):
        self.model_id = model_id
        self.idle_timeout = idle_timeout
        self.always_loaded = always_loaded
        self.last_used: float = time.time()
        self.loaded: bool = True   # assume loaded at start
        self.active_requests: int = 0

    def touch(self):
        self.last_used = time.time()

    def idle_seconds(self) -> float:
        return time.time() - self.last_used


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class AdminClient:
    def __init__(self, base_url: str, token: str = ""):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30)

    async def get_queue_stats(self, model_id: str) -> dict[str, Any] | None:
        """GET /v1/admin/models/{model_id}/stats"""
        try:
            r = await self._client.get(f"/v1/admin/models/{model_id}/stats")
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None   # model not loaded
        except Exception as e:
            log.debug(f"stats error for {model_id}: {e}")
        return {}

    async def unload_model(self, model_id: str) -> bool:
        """POST /v1/admin/models/{model_id}/unload"""
        try:
            r = await self._client.post(f"/v1/admin/models/{model_id}/unload")
            return r.status_code == 200
        except Exception as e:
            log.error(f"unload error for {model_id}: {e}")
            return False

    async def load_model(self, model_id: str) -> bool:
        """POST /v1/admin/models/{model_id}/load"""
        try:
            r = await self._client.post(f"/v1/admin/models/{model_id}/load")
            return r.status_code == 200
        except Exception as e:
            log.error(f"load error for {model_id}: {e}")
            return False

    async def list_loaded_models(self) -> list[str]:
        """GET /v1/models"""
        try:
            r = await self._client.get("/v1/models")
            if r.status_code == 200:
                data = r.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            log.debug(f"list models error: {e}")
        return []

    async def aclose(self):
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Main watchdog loop
# ---------------------------------------------------------------------------

class IdleUnloadWatchdog:
    def __init__(self, cfg: WatchdogConfig):
        self.cfg = cfg
        self.client = AdminClient(cfg.base_url, cfg.admin_token)
        self.states: dict[str, ModelState] = {
            m.model_id: ModelState(m.model_id, m.idle_timeout, m.always_loaded)
            for m in cfg.models
        }

    async def run(self):
        log.info(f"Watchdog started. base_url={self.cfg.base_url}, "
                 f"check_interval={self.cfg.check_interval}s")
        log.info(f"Watching {len(self.states)} models: {list(self.states.keys())}")

        while True:
            try:
                await self._check_all()
            except Exception as e:
                log.error(f"Watchdog check failed: {e}")
            await asyncio.sleep(self.cfg.check_interval)

    async def _check_all(self):
        # 先获取当前已加载的模型列表
        loaded = set(await self.client.list_loaded_models())

        for model_id, state in self.states.items():
            state.loaded = model_id in loaded

            if state.always_loaded:
                # 确保 always_loaded 的模型始终在线
                if not state.loaded:
                    log.info(f"[{model_id}] always_loaded=true, reloading...")
                    ok = await self.client.load_model(model_id)
                    if ok:
                        state.loaded = True
                        state.touch()
                        log.info(f"[{model_id}] reloaded ✓")
                continue

            if state.idle_timeout <= 0:
                continue  # 不管理这个模型

            if not state.loaded:
                # 已经 unload，不需要处理
                log.debug(f"[{model_id}] already unloaded, skipping")
                continue

            # 查询活跃请求数
            stats = await self.client.get_queue_stats(model_id)
            if stats is None:
                # 404 = 已经不在 registry 里了
                state.loaded = False
                log.info(f"[{model_id}] not found in registry (already unloaded)")
                continue

            active = stats.get("queue_stats", {}).get("active_requests", 0)
            if active > 0:
                state.touch()
                log.debug(f"[{model_id}] active_requests={active}, resetting idle timer")
                continue

            idle = state.idle_seconds()
            log.debug(f"[{model_id}] idle={idle:.0f}s / timeout={state.idle_timeout}s")

            if idle >= state.idle_timeout:
                log.info(f"[{model_id}] idle {idle:.0f}s >= {state.idle_timeout}s, unloading...")
                ok = await self.client.unload_model(model_id)
                if ok:
                    state.loaded = False
                    log.info(f"[{model_id}] unloaded ✓ (freed memory)")
                else:
                    log.warning(f"[{model_id}] unload failed")

    async def aclose(self):
        await self.client.aclose()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="MLX Server Idle Unload Watchdog")
    parser.add_argument(
        "--watchdog-config",
        default=str(Path(__file__).parent / "watchdog.yaml"),
        help="Path to watchdog config YAML",
    )
    args = parser.parse_args()

    cfg = load_watchdog_config(args.watchdog_config)
    watchdog = IdleUnloadWatchdog(cfg)

    try:
        await watchdog.run()
    except KeyboardInterrupt:
        log.info("Watchdog stopped")
    finally:
        await watchdog.aclose()


if __name__ == "__main__":
    asyncio.run(main())
