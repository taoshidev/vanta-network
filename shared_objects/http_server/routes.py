"""Generic route registration so adding a new signed-request endpoint is one call.

Each route: authorize (registered-hotkey check, replacing bt.Axon's blacklist_fn) ->
offload the actual handler onto the shared thread pool (replacing the
`loop.run_in_executor(self._thread_pool, ...)` pattern every handler in
neurons/validator_base.py already used) -> return the response model.

`authz_check` is a plain `Callable[[str], None]` (raise to reject) rather than a hardcoded
`metagraph_client.has_hotkey(...)` RPC call: the old bt.Axon-era `blacklist_fn` on the
`Validator` subclass (neurons/validator.py) answered this from an in-process cache
(subtensor_ops_manager's registered-hotkey set, or a background-refreshed local set in
the orders-app split) specifically to avoid an RPC round-trip on every single signed
request — a real latency difference under concurrent order load, not a cosmetic one.
Callers should pass that fast check in where available (see
neurons/validator_base.py::wire_http_server, which defaults to
shared_objects.http_auth.require_registered_hotkey — an RPC call — only when no faster
check is provided)."""
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

    async def endpoint(body: request_model, hotkey: str = Depends(caller_hotkey)):
        try:
            authz_check(hotkey)
        except AuthError as e:
            raise HTTPException(status_code=403, detail=str(e))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(thread_pool, handler, body, hotkey)

    app.add_api_route(f"/{path}", endpoint, methods=["POST"], response_model=response_model)
