#!/usr/bin/env python3
"""
MLX Server 启动入口，注入 lazy loading + admin API。
替代: mlx-openai-server launch --config config.yaml
"""
import sys
sys.path.insert(0, "/Users/ben/.mlx-server")

if __name__ == "__main__":
    import dataclasses
    import gc
    from contextlib import asynccontextmanager

    import mlx.core as mx
    import yaml
    from fastapi import FastAPI
    from loguru import logger

    import app.server as _server_mod
    from app.core.handler_process import HandlerProcessProxy
    from app.core.model_registry import ModelRegistry

    import admin_api_patch
    from lazy_handler_proxy import LazyHandlerProxy

    _CONFIG_PATH = "/Users/ben/.mlx-server/config.yaml"

    def _load_lazy_flags() -> dict[str, dict]:
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        return {
            m["model_id"]: {
                "lazy": m.get("lazy", False),
                "idle_timeout": m.get("idle_timeout", 1800),
            }
            for m in raw.get("models", [])
            if "model_id" in m
        }

    def _patched_multi_lifespan(config):
        lazy_flags = _load_lazy_flags()

        @asynccontextmanager
        async def lifespan(application: FastAPI):
            import asyncio
            registry = ModelRegistry()
            lazy_proxies: dict[str, tuple] = {}

            try:
                for model_cfg in config.models:
                    model_id = model_cfg.model_id
                    flags = lazy_flags.get(model_id, {})
                    is_lazy = flags.get("lazy", False)

                    proxy = HandlerProcessProxy(
                        model_cfg_dict=dataclasses.asdict(model_cfg),
                        model_type=model_cfg.model_type,
                        model_path=model_cfg.model_path,
                        model_id=model_id,
                    )
                    queue_config = {
                        "max_concurrency": model_cfg.max_concurrency,
                        "timeout": model_cfg.queue_timeout,
                        "queue_size": model_cfg.queue_size,
                    }

                    if is_lazy:
                        handler = LazyHandlerProxy(proxy, queue_config)
                        idle_timeout = flags.get("idle_timeout", 1800)
                        lazy_proxies[model_id] = (handler, idle_timeout)
                        logger.info(f"[lazy] Registered '{model_id}' (idle_timeout={idle_timeout}s, not loaded yet)")
                    else:
                        await proxy.start(queue_config)
                        handler = proxy
                        logger.info(f"[eager] Loaded '{model_id}'")

                    await registry.register_model(
                        model_id=model_id,
                        handler=handler,
                        model_type=model_cfg.model_type,
                        context_length=model_cfg.context_length,
                    )

                application.state.registry = registry
                application.state.lazy_proxies = lazy_proxies
                if config.models:
                    application.state.handler = registry.get_handler(config.models[0].model_id)

                admin_api_patch.install(application)

                logger.info("[start_with_admin] Startup complete ✓")

            except Exception as e:
                logger.error(f"Startup failed: {e}")
                await registry.cleanup_all()
                raise

            mx.clear_cache()
            gc.collect()
            yield

            logger.info("[start_with_admin] Shutting down")
            await registry.cleanup_all()
            mx.clear_cache()
            gc.collect()

        return lifespan

    # Patch must happen before cli() is called
    _server_mod.create_multi_lifespan = _patched_multi_lifespan

    from app.cli import cli
    sys.exit(cli())
