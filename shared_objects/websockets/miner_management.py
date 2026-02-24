from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from shared_objects.websockets.base import (
    WebSocketConnectionSettings,
    WebSocketConnectionTemplate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinerWebSocketMessage:
    miner_hotkey: str
    message: str | bytes


class WebSocketMinerConnectionPool:
    """
    Queue-driven "multi-threaded" (concurrent) pool for many miner websocket connections.

    This is asyncio-concurrent (tasks), not OS threads (which is the right fit for `websockets`).

    - Feed miner hotkeys into `set_targets()` (or `add_target()` / `remove_target()`)
    - A worker loop reconciles desired targets with active connections
    - Each miner gets its own `WebSocketConnectionTemplate` that auto-reconnects + keeps alive
    - Inbound messages are pushed into a single pool inbox as `MinerWebSocketMessage`
    """

    def __init__(
        self,
        *,
        connection_builder: Callable[[str], WebSocketConnectionTemplate],
        n_workers: int = 4,
        inbox_maxsize: int = 0,
    ):
        self._build_connection = connection_builder

        self._cmd_q: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._stop = asyncio.Event()

        self._targets: set[str] = set()
        self._connections: dict[str, WebSocketConnectionTemplate] = {}
        self._tasks: dict[str, asyncio.Task] = {}

        self._inbox: asyncio.Queue[MinerWebSocketMessage] = asyncio.Queue(maxsize=inbox_maxsize)
        self._n_workers = max(1, n_workers)

    @property
    def targets(self) -> set[str]:
        return set(self._targets)

    @property
    def active(self) -> set[str]:
        return {hk for hk, t in self._tasks.items() if not t.done()}

    async def start(self) -> None:
        if self._workers:
            return
        self._stop.clear()
        self._workers = [asyncio.create_task(self._worker_loop()) for _ in range(self._n_workers)]

    async def stop(self) -> None:
        self._stop.set()

        # Stop all connections
        for hk, conn in list(self._connections.items()):
            try:
                conn.stop()
            except Exception:
                logger.debug("stop() failed for %s", hk, exc_info=True)

        # Cancel worker loops
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            try:
                await w
            except asyncio.CancelledError:
                pass
        self._workers.clear()

        # Await connection tasks
        for t in list(self._tasks.values()):
            try:
                await t
            except Exception:
                pass
        self._tasks.clear()
        self._connections.clear()
        self._targets.clear()

    async def recv(self, *, timeout_s: Optional[float] = None) -> MinerWebSocketMessage:
        if timeout_s is None:
            return await self._inbox.get()
        return await asyncio.wait_for(self._inbox.get(), timeout=timeout_s)

    async def add_target(self, miner_hotkey: str) -> None:
        self._targets.add(miner_hotkey)
        await self._cmd_q.put(("add", miner_hotkey))

    async def remove_target(self, miner_hotkey: str) -> None:
        self._targets.discard(miner_hotkey)
        await self._cmd_q.put(("remove", miner_hotkey))

    async def set_targets(self, miner_hotkeys: set[str]) -> None:
        desired = set(miner_hotkeys)
        to_add = desired - self._targets
        to_remove = self._targets - desired

        self._targets = desired

        for hk in to_add:
            await self._cmd_q.put(("add", hk))
        for hk in to_remove:
            await self._cmd_q.put(("remove", hk))

    async def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                action, hotkey = await self._cmd_q.get()
            except asyncio.CancelledError:
                return

            try:
                if action == "add":
                    await self._ensure_connection(hotkey)
                elif action == "remove":
                    await self._teardown_connection(hotkey)
            finally:
                self._cmd_q.task_done()

    async def _ensure_connection(self, hotkey: str) -> None:
        existing = self._tasks.get(hotkey)
        if existing is not None and not existing.done():
            return

        conn = self._connections.get(hotkey)
        if conn is None:
            conn = self._build_connection(hotkey)
            self._connections[hotkey] = conn

        async def on_message(_ws, msg: str | bytes) -> None:
            # fan-in to one queue with hotkey tagging
            await self._inbox.put(MinerWebSocketMessage(miner_hotkey=hotkey, message=msg))

        # Wrap/override message handling to ensure pool always receives messages
        original_on_message = getattr(conn, "_on_message", None)

        async def combined_on_message(ws, msg: str | bytes) -> None:
            await on_message(ws, msg)
            if original_on_message is not None:
                await original_on_message(ws, msg)

        conn._on_message = combined_on_message  # noqa: SLF001 (intentional internal hook)

        self._tasks[hotkey] = asyncio.create_task(conn.run_forever())

    async def _teardown_connection(self, hotkey: str) -> None:
        conn = self._connections.pop(hotkey, None)
        task = self._tasks.pop(hotkey, None)

        if conn is not None:
            conn.stop()

        if task is not None and not task.done():
            try:
                await task
            except Exception:
                pass


def default_connection_builder(
    miner_hotkey: str,
    *,
    settings: Optional[WebSocketConnectionSettings] = None,
    build_auth_message: Optional[Callable[[str], dict[str, Any] | str | bytes]] = None,
) -> WebSocketConnectionTemplate:
    """
    Convenience builder when each miner uses the same websocket endpoint.
    If you need per-miner endpoints, provide your own `connection_builder`.
    """
    if settings is None:
        settings = WebSocketConnectionTemplate.settings_from_vali_config()

    def _auth():
        if build_auth_message is None:
            return {}
        return build_auth_message(miner_hotkey)

    return WebSocketConnectionTemplate(settings=settings, build_auth_message=_auth)