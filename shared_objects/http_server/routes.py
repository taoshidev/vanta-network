"""Generic route registration so adding a new signed-request endpoint is one call.

Each route: authorize (registered-hotkey check) -> offload the handler onto the shared
thread pool -> return the response model.

`authz_check` is a plain `Callable[[str], None]` (raise to reject) rather than a hardcoded
RPC call, since an in-process cache (e.g. subtensor_ops_manager's registered-hotkey set)
avoids an RPC round-trip on every signed request — a real latency difference under
concurrent order load. Callers should pass that fast check in where available (see
neurons/validator_base.py::wire_http_server, which defaults to the slower
shared_objects.http_auth.require_registered_hotkey only when no faster check is given)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Type

from fastapi import Depends, HTTPException
from pydantic import BaseModel

from shared_objects.http_auth.exceptions import AuthError
from shared_objects.http_server.app import caller_hotkey


def register_handler(
    app,
    path: str,
    request_model: Type[BaseModel],
    response_model: Type[BaseModel],
    handler: Callable[[BaseModel, str], BaseModel],
    thread_pool: ThreadPoolExecutor,
    authz_check: Callable[[str], None],
) -> None:
    """Registers POST /{path}. `handler(body, caller_hotkey) -> response_model` runs in
    thread_pool; `authz_check(caller_hotkey)` (raise to reject) runs first, on the event loop."""

    async def endpoint(body: request_model, hotkey: str = Depends(caller_hotkey)):  # pyright: ignore[reportInvalidTypeForm]
        try:
            authz_check(hotkey)
        except AuthError as e:
            raise HTTPException(status_code=403, detail=str(e))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(thread_pool, handler, body, hotkey)

    app.add_api_route(f"/{path}", endpoint, methods=["POST"], response_model=response_model)
