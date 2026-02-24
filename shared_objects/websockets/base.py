from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from vali_objects.vali_config import ValiConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSocketConnectionSettings:
    host: str
    port: int
    secure: bool = False
    path: str = "/"

    connect_timeout_s: float = 10.0
    close_timeout_s: float = 5.0

    # WebSocket protocol keepalive (built-in ping frames)
    ping_interval_s: float = 30.0
    ping_timeout_s: float = 10.0

    # If no message is received for this long, perform a liveness check
    # (send a ping and require a pong). This helps detect stalled connections
    # even when the server is quiet.
    recv_timeout_s: Optional[float] = 60.0

    # Timeout for the liveness ping/pong exchange triggered by recv_timeout_s.
    liveness_ping_timeout_s: float = 5.0

    # Application-level keepalive (optional; set to None to disable)
    app_ping_interval_s: Optional[float] = None

    # Reconnect/backoff
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1

    max_message_size: Optional[int] = None  # None = unlimited
    compression: Optional[str] = "deflate"

    def uri(self) -> str:
        protocol = "wss" if self.secure else "ws"
        path = self.path if self.path.startswith("/") else f"/{self.path}"
        return f"{protocol}://{self.host}:{self.port}{path}"


class WebSocketConnectionTemplate:
    """
    Generic WebSocket client connection manager:
    - Reads default host/port from `ValiConfig` (overrideable)
    - Maintains a connection loop with exponential backoff + jitter
    - Supports protocol keepalive (ping frames) and optional app-level ping
    - Dispatches received messages to an async callback

    Intended usage:
        settings = WebSocketConnectionTemplate.settings_from_vali_config()
        client = WebSocketConnectionTemplate(
            settings=settings,
            on_open=my_on_open,
            on_message=my_on_message,
            on_close=my_on_close,
            on_error=my_on_error,
            build_auth_message=my_auth_builder,
        )
        await client.run_forever()
    """

    def __init__(
        self,
        *,
        settings: WebSocketConnectionSettings,
        on_open: Optional[Callable[[WebSocketClientProtocol], Awaitable[None]]] = None,
        on_message: Optional[
            Callable[[WebSocketClientProtocol, str | bytes], Awaitable[None]]
        ] = None,
        on_close: Optional[Callable[[Optional[BaseException]], Awaitable[None]]] = None,
        on_error: Optional[Callable[[BaseException], Awaitable[None]]] = None,
        build_auth_message: Optional[Callable[[], dict[str, Any] | str | bytes]] = None,
        build_app_ping_message: Optional[Callable[[], dict[str, Any] | str | bytes]] = None,
    ):
        self.settings = settings

        self._on_open = on_open
        self._on_message = on_message
        self._on_close = on_close
        self._on_error = on_error

        self._build_auth_message = build_auth_message
        self._build_app_ping_message = build_app_ping_message

        self._stop_event = asyncio.Event()
        self._ws: Optional[WebSocketClientProtocol] = None
        self._app_ping_task: Optional[asyncio.Task] = None
        self._inbox: asyncio.Queue[str | bytes] = asyncio.Queue()
        self._runner_task: Optional[asyncio.Task] = None

    @staticmethod
    def settings_from_vali_config(
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        secure: bool = False,
        path: str = "/",
        **overrides: Any,
    ) -> WebSocketConnectionSettings:
        """
        Create settings using `ValiConfig.VANTA_WEBSOCKET_HOST/PORT` as defaults.
        Any keyword in WebSocketConnectionSettings can be overridden via `overrides`.
        """
        base = WebSocketConnectionSettings(
            host=host if host is not None else ValiConfig.VANTA_WEBSOCKET_HOST,
            port=port if port is not None else ValiConfig.VANTA_WEBSOCKET_PORT,
            secure=secure,
            path=path,
        )
        if not overrides:
            return base
        return WebSocketConnectionSettings(**{**base.__dict__, **overrides})

    @property
    def websocket(self) -> Optional[WebSocketClientProtocol]:
        return self._ws

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    def start_in_background(self) -> asyncio.Task:
        """
        Start `run_forever()` as a background task.
        Useful when you want the client to continuously receive data while your
        main coroutine does other work.
        """
        if self._runner_task is not None and not self._runner_task.done():
            return self._runner_task
        self._runner_task = asyncio.create_task(self.run_forever())
        return self._runner_task

    async def wait_closed(self) -> None:
        if self._runner_task is not None:
            await self._runner_task

    async def stop_and_wait(self) -> None:
        self.stop()
        await self.wait_closed()

    def stop(self) -> None:
        """Signal the connection loop to stop and close the websocket."""
        self._stop_event.set()
        if self._ws is not None:
            try:
                asyncio.get_running_loop().create_task(self._ws.close())
            except RuntimeError:
                # No running loop (caller may be stopping from sync context)
                pass

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self.send_text(json.dumps(payload))

    async def send_text(self, text: str) -> None:
        if not self.is_connected or self._ws is None:
            raise ConnectionError("WebSocket is not connected")
        await self._ws.send(text)

    async def send_bytes(self, data: bytes) -> None:
        if not self.is_connected or self._ws is None:
            raise ConnectionError("WebSocket is not connected")
        await self._ws.send(data)

    async def run_forever(self) -> None:
        """
        Main connection loop. Reconnects until `stop()` is called.
        """
        backoff = self.settings.initial_backoff_s

        while not self._stop_event.is_set():
            exc: Optional[BaseException] = None
            try:
                async with websockets.connect(
                    self.settings.uri(),
                    compression=self.settings.compression,
                    open_timeout=self.settings.connect_timeout_s,
                    close_timeout=self.settings.close_timeout_s,
                    ping_interval=self.settings.ping_interval_s,
                    ping_timeout=self.settings.ping_timeout_s,
                    max_size=self.settings.max_message_size,
                ) as ws:
                    self._ws = ws

                    # Reset backoff after a successful connection
                    backoff = self.settings.initial_backoff_s

                    if self._build_auth_message is not None:
                        await self._send_any(self._build_auth_message())

                    if self._on_open is not None:
                        await self._on_open(ws)

                    # Optional application-level ping loop
                    self._app_ping_task = None
                    if (
                        self.settings.app_ping_interval_s is not None
                        and self._build_app_ping_message is not None
                    ):
                        self._app_ping_task = asyncio.create_task(self._app_ping_loop())

                    await self._recv_loop(ws)

            except asyncio.CancelledError:
                raise
            except BaseException as e:
                exc = e
                if self._on_error is not None and isinstance(e, Exception):
                    try:
                        await self._on_error(e)
                    except Exception:
                        logger.exception("WebSocket on_error callback failed")
                else:
                    logger.warning("WebSocket connection error: %s", e)
            finally:
                if self._app_ping_task is not None:
                    self._app_ping_task.cancel()
                    try:
                        await self._app_ping_task
                    except asyncio.CancelledError:
                        pass
                    self._app_ping_task = None

                self._ws = None

                if self._on_close is not None:
                    try:
                        await self._on_close(exc)
                    except Exception:
                        logger.exception("WebSocket on_close callback failed")

            if self._stop_event.is_set():
                break

            await asyncio.sleep(self._with_jitter(backoff))
            backoff = min(backoff * self.settings.backoff_multiplier, self.settings.max_backoff_s)

    async def _recv_loop(self, ws: WebSocketClientProtocol) -> None:
        while not self._stop_event.is_set():
            try:
                if self.settings.recv_timeout_s is None:
                    msg = await ws.recv()
                else:
                    msg = await asyncio.wait_for(ws.recv(), timeout=self.settings.recv_timeout_s)
            except asyncio.TimeoutError:
                # No data received recently; verify connection is still alive.
                try:
                    pong_waiter = await ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=self.settings.liveness_ping_timeout_s)
                    continue
                except Exception:
                    return
            except websockets.exceptions.ConnectionClosed:
                return

            # Always provide a default way to consume messages, even if caller
            # doesn't pass an on_message callback.
            await self._inbox.put(msg)

            if self._on_message is not None:
                await self._on_message(ws, msg)

    async def recv(self, *, timeout_s: Optional[float] = None) -> str | bytes:
        """
        Receive the next message from the connection.
        This works whether you pass `on_message` or not.
        """
        if timeout_s is None:
            return await self._inbox.get()
        return await asyncio.wait_for(self._inbox.get(), timeout=timeout_s)

    async def iter_messages(self):
        """Async generator of inbound messages."""
        while not self._stop_event.is_set():
            yield await self.recv()

    async def _app_ping_loop(self) -> None:
        assert self.settings.app_ping_interval_s is not None
        assert self._build_app_ping_message is not None

        while not self._stop_event.is_set():
            await asyncio.sleep(self.settings.app_ping_interval_s)
            if not self.is_connected or self._ws is None:
                continue
            try:
                await self._send_any(self._build_app_ping_message())
            except Exception:
                # Any exception here should be handled by the main loop when recv fails;
                # but don't kill the task noisily.
                logger.debug("App ping send failed", exc_info=True)

    async def _send_any(self, data: dict[str, Any] | str | bytes) -> None:
        if isinstance(data, dict):
            await self.send_json(data)
        elif isinstance(data, str):
            await self.send_text(data)
        elif isinstance(data, bytes):
            await self.send_bytes(data)
        else:
            raise TypeError(f"Unsupported outbound message type: {type(data)}")

    def _with_jitter(self, backoff_s: float) -> float:
        if self.settings.jitter_ratio <= 0:
            return backoff_s
        jitter = random.uniform(-self.settings.jitter_ratio, self.settings.jitter_ratio)
        return max(0.0, backoff_s * (1.0 + jitter))

