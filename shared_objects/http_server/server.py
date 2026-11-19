"""Owns the uvicorn.Server lifecycle for the signed-request FastAPI app.

Mirrors the start()/stop() shape of bittensor v10's bt.Axon so validator_base.py's
process-split call sites (--no-axon / --orders-app, see neurons/validator_base.py) need
minimal changes beyond constructing an HttpServer instead of a bt.Axon.
"""
import threading

import uvicorn

from shared_objects.log import logger


class HttpServer:
    def __init__(self, app, host: str, port: int):
        self.host = host
        self.port = port
        self._config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
        self._server = uvicorn.Server(self._config)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        logger.info(f"[http_server] starting on {self.host}:{self.port}")
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="http-server")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
