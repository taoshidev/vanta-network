"""httpx-based signed-request client. `aquery` fans out to every target concurrently via
asyncio.gather — the multi-validator fire-and-forget broadcast relies on this not being
sequential. Target objects just need `.ip`, `.port`, `.hotkey` (the SimpleAxonInfo shape
MetagraphClient returns).

Constructs a fresh httpx.AsyncClient per instance, so a latency-sensitive call site
issuing many sequential requests to the same targets should hold one HttpAuthClient for
its lifetime to reuse connections, rather than building one per request.
"""
import asyncio
from typing import Optional, Type

import httpx
from pydantic import BaseModel

from shared_objects.http_client.auth import HotkeyAuth
from shared_objects.log import logger


def _failure_response(response_model: Type[BaseModel], error_message: str) -> BaseModel:
    kwargs = {"error_message": error_message}
    fields = response_model.model_fields
    if "successfully_processed" in fields:
        kwargs["successfully_processed"] = False
    if "should_retry" in fields:
        kwargs["should_retry"] = True
    return response_model(**kwargs)


class HttpAuthClient:
    def __init__(self, wallet, timeout: float = 12.0):
        self.wallet = wallet
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "HttpAuthClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _post_one(self, target, request_name: str, payload: bytes,
                         response_model: Type[BaseModel], timeout: Optional[float]):
        url = f"http://{target.ip}:{target.port}/{request_name}"
        try:
            r = await self._client.post(
                url, content=payload,
                headers={"Content-Type": "application/json"},
                auth=HotkeyAuth(self.wallet, receiver_ss58=target.hotkey),
                timeout=timeout if timeout is not None else self.timeout,
            )
            if r.status_code == 200:
                return response_model.model_validate_json(r.content)
            return _failure_response(response_model, f"HTTP {r.status_code}: {r.text}")
        except Exception as e:
            logger.debug(f"[http_client] {request_name} to {getattr(target, 'hotkey', '?')} "
                         f"({getattr(target, 'ip', '?')}:{getattr(target, 'port', '?')}) failed: {e}")
            return _failure_response(response_model, str(e))

    async def aquery(
        self,
        targets: list,
        request_name: str,
        body: BaseModel,
        response_model: Type[BaseModel],
        timeout: Optional[float] = None,
    ) -> list[BaseModel]:
        """Sends `body` to every target concurrently; returns one response_model per
        target, in the same order as `targets` (zip them together at the call site to
        know which target a given response came from)."""
        payload = body.model_dump_json().encode()
        return await asyncio.gather(*[
            self._post_one(target, request_name, payload, response_model, timeout)
            for target in targets
        ])
