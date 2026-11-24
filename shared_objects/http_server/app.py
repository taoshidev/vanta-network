"""FastAPI app factory for the miner<->validator / validator<->validator signed-request
server, replacing bittensor v10's bt.Axon (which was itself a Starlette app under the
hood — this is the same ASGI stack, just owned directly instead of hidden behind bt.Axon).

Authentication (dual-scheme signature verification) runs as Starlette middleware, mirroring
bittensor v10's AxonMiddleware.dispatch(preprocess -> verify) sequencing, so it rejects an
unauthenticated caller before any route handler or registered-hotkey check runs.
Authorization (is this verified hotkey welcome here) stays a separate per-route concern
via the `caller_hotkey` dependency below plus shared_objects.http_auth.require_registered_hotkey
— this replaces bt.Axon's per-route blacklist_fn.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from fastapi import FastAPI

from shared_objects.http_auth import AuthErrorTypes, AuthVerifier
from shared_objects.log import logger


class BtAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, verifier: AuthVerifier, legacy_hash_fields_by_path: dict):
        super().__init__(app)
        self.verifier = verifier
        self.legacy_hash_fields_by_path = legacy_hash_fields_by_path

    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        message_name = request.url.path.lstrip("/")

        # ASGI's `path` is percent-decoded; signatures cover the wire (percent-encoded)
        # form, so rebuild it from raw_path + query_string per SIGNED-REQUESTS.md's
        # FastAPI recipe. Our routes are fixed ASCII paths with no query string in
        # practice, but this keeps the verifier's path check byte-exact regardless.
        target = request.scope["raw_path"].decode()
        if request.scope["query_string"]:
            target += "?" + request.scope["query_string"].decode()

        try:
            caller = self.verifier.verify(
                request.headers, body,
                method=request.method, path=target,
                message_name=message_name,
                legacy_hash_fields=self.legacy_hash_fields_by_path.get(message_name),
            )
        except AuthErrorTypes as e:
            logger.warning(f"[http_server] rejected {request.method} {request.url.path} "
                            f"from {request.client}: {type(e).__name__}: {e}")
            return JSONResponse({"error": str(e)}, status_code=401)

        request.state.caller_hotkey = caller.hotkey_ss58
        request.state.caller_scheme = caller.scheme
        return await call_next(request)


def caller_hotkey(request: Request) -> str:
    """FastAPI dependency: the authenticated caller's hotkey, set by BtAuthMiddleware."""
    return request.state.caller_hotkey


def build_app(verifier: AuthVerifier, legacy_hash_fields_by_path: dict) -> FastAPI:
    app = FastAPI()
    app.add_middleware(BtAuthMiddleware, verifier=verifier, legacy_hash_fields_by_path=legacy_hash_fields_by_path)
    return app
