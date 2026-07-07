"""
Readiness watchdog for the standalone API apps (vanta-rest / vanta-ws).

Both standalone servers are designed to be TOLERANT of core being slow or absent at
startup: they boot, bind their front door, and connect to core's state servers lazily
(retrying per use) rather than crashing. That tolerance is good, but it can mask a
genuinely stuck server — one that came up but can never reach core. This watchdog makes
the tolerance OBSERVABLE: it alerts (via the existing SlackNotifier) if the app fails to
become healthy within a grace window, and again when it recovers.

Design notes:
  - "Healthy" = the app's own front door is bound AND core's state tier is reachable. We
    use a cheap bounded TCP probe (no RPC handshake, no 60s client-retry block) so the
    watchdog loop stays fast even while core is down.
  - Core-liveness is proxied by PositionManager (:50002), the central state server both
    apps depend on. This means "core reachable", not "every downstream server up" — the
    right altitude for a spin-up alert (probing every dependency would flap).
  - Edge-triggered: at most one alert per unhealthy episode and one on recovery, so a
    normal startup race does not spam. SlackNotifier's own 300s cooldown is a backstop;
    the transition alerts pass bypass_cooldown=True so they are never silently dropped.
  - Never crashes the app; it only observes and alerts.
"""

import socket
import threading
import time

import bittensor as bt


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    # 0.0.0.0 is a bind address, not a connect address — probe loopback for local ports.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", None) else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _safe_alert(slack_notifier, message: str, level: str) -> None:
    # Transition alerts are rare (edge-triggered), so bypass the cooldown to guarantee delivery.
    try:
        if slack_notifier is not None:
            slack_notifier.send_message(message, level=level, bypass_cooldown=True)
    except Exception as e:
        bt.logging.warning(f"[readiness] Slack alert failed: {e}")
    bt.logging.info(f"[readiness] {message}")


def start_readiness_watchdog(
    *,
    app_name: str,
    slack_notifier,
    front_door_host: str,
    front_door_port: int,
    core_probe_ports,
    grace_s: float = 45.0,
    check_interval_s: float = 5.0,
    stop_event: threading.Event = None,
) -> threading.Thread:
    """
    Start a daemon thread that alerts on sustained unhealthiness and on recovery.

    Args:
        app_name: e.g. "vanta-rest" / "vanta-ws" (used in alert text + thread name).
        slack_notifier: SlackNotifier (may be None — then alerts are log-only).
        front_door_host / front_door_port: the app's public listener to confirm it is bound.
        core_probe_ports: iterable of core RPC ports whose reachability means "core is up".
        grace_s: how long to tolerate un-readiness before the first alert (covers spin-up).
        check_interval_s: seconds between probes.
        stop_event: optional; set on shutdown so the watchdog exits promptly.

    Returns the started (daemon) thread.
    """
    core_probe_ports = list(core_probe_ports)

    def _healthy() -> bool:
        if not _port_open(front_door_host, front_door_port):
            return False
        return all(_port_open("127.0.0.1", p) for p in core_probe_ports)

    def _loop() -> None:
        start = time.time()
        alerted_unhealthy = False
        logged_ready = False
        while stop_event is None or not stop_event.is_set():
            if _healthy():
                if not logged_ready:
                    bt.logging.success(
                        f"[{app_name}] readiness: HEALTHY (front door :{front_door_port} bound, "
                        f"core reachable on {core_probe_ports})"
                    )
                    logged_ready = True
                if alerted_unhealthy:
                    _safe_alert(slack_notifier, f"✅ {app_name} recovered — front door bound and core reachable.", "info")
                    alerted_unhealthy = False
            else:
                elapsed = time.time() - start
                if elapsed >= grace_s and not alerted_unhealthy:
                    fd = _port_open(front_door_host, front_door_port)
                    core_status = {p: _port_open("127.0.0.1", p) for p in core_probe_ports}
                    _safe_alert(
                        slack_notifier,
                        f"🔴 {app_name} not healthy after {int(elapsed)}s "
                        f"(front door :{front_door_port} bound={fd}; core ports {core_status}). "
                        f"Process is up and will keep retrying — check that vanta-core is running.",
                        "error",
                    )
                    alerted_unhealthy = True

            if stop_event is not None:
                if stop_event.wait(check_interval_s):
                    break
            else:
                time.sleep(check_interval_s)

    t = threading.Thread(target=_loop, name=f"{app_name}-readiness", daemon=True)
    t.start()
    return t
