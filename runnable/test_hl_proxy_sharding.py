#!/usr/bin/env python3
"""
test_hl_proxy_sharding.py - End-to-end test for Hyperliquid WebSocket proxy sharding.

Standalone script (no Vanta validator dependencies). Reads proxy config from
secrets.json, discovers active wallets from the Hyperliquid REST/WS API, then
spins up multiple sharded WebSocket connections through different Decodo
SOCKS5 proxy IPs and watches those wallets for userFills.

Uses python-socks to create SOCKS5 tunnels, compatible with websockets <=14.x.

Usage:
    python runnable/test_hl_proxy_sharding.py [--wallets N] [--watch-seconds S] [--discovery-seconds S]

Phases:
    1. Load proxy config from secrets.json (hl_proxy_url, hl_proxy_ports)
    2. Verify each proxy port gets a distinct external IP (via httpbin)
    3. Discover active wallets via HL REST API + WS trades feed
    4. Distribute wallets across shards (1 shard per proxy port, max 10 each)
    5. Subscribe each shard to userFills for its assigned wallets
    6. Watch for fills, print them live, and report summary

Requirements:
    pip install websockets python-socks requests
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import ssl
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Set

try:
    import websockets
except ImportError:
    sys.exit("ERROR: websockets not installed. Run: pip install websockets")

try:
    from python_socks.async_.asyncio import Proxy
except ImportError:
    sys.exit("ERROR: python-socks not installed. Run: pip install python-socks")

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests not installed. Run: pip install requests")


# ─── Constants ───────────────────────────────────────────────────────────────

HL_WS_URL = "wss://api.hyperliquid.xyz/ws"
HL_WS_HOST = "api.hyperliquid.xyz"
HL_WS_PORT = 443
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
MAX_ADDRS_PER_SHARD = 10
HEARTBEAT_INTERVAL_S = 30.0
RECONNECT_BACKOFF_MAX_S = 30.0
MAX_CONSECUTIVE_FAILURES = 5

# ANSI colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_CYAN = "\033[36m"
C_DIM = "\033[2m"


# ─── Proxy helpers ───────────────────────────────────────────────────────────

def parse_proxy_url(proxy_url: str) -> tuple:
    """Parse 'socks5://user:pass@host' into (username, password, host)."""
    m = re.match(r'socks5://(?:([^:]+):([^@]+)@)?(.+)', proxy_url)
    if not m:
        raise ValueError(f"Cannot parse proxy URL: {proxy_url}")
    return m.group(1), m.group(2), m.group(3)


async def create_proxy_socket(proxy_url: str, port: int, dest_host: str, dest_port: int):
    """Create a SOCKS5-tunneled socket to dest_host:dest_port via proxy_url:port."""
    username, password, host = parse_proxy_url(proxy_url)
    proxy = Proxy.from_url(f"socks5://{username}:{password}@{host}:{port}")
    sock = await asyncio.wait_for(
        proxy.connect(dest_host=dest_host, dest_port=dest_port),
        timeout=15,
    )
    return sock


async def connect_ws_proxied(proxy_url: str, port: int):
    """Open a WebSocket to HL through a SOCKS5 proxy. Returns a ws connection context."""
    sock = await create_proxy_socket(proxy_url, port, HL_WS_HOST, HL_WS_PORT)
    ssl_ctx = ssl.create_default_context()
    return websockets.connect(
        HL_WS_URL,
        sock=sock,
        ssl=ssl_ctx,
        server_hostname=HL_WS_HOST,
        ping_interval=None,
        close_timeout=5,
    )


async def connect_ws_direct():
    """Open a direct (no proxy) WebSocket to HL."""
    return websockets.connect(HL_WS_URL, ping_interval=None, close_timeout=5)


# ─── Secrets loading ─────────────────────────────────────────────────────────

def load_secrets() -> dict:
    """Load secrets.json from the repo root."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    secrets_path = os.path.join(base_dir, "secrets.json")
    if not os.path.exists(secrets_path):
        print(f"{C_RED}ERROR: secrets.json not found at {secrets_path}{C_RESET}")
        print("Create it with at least:")
        print(json.dumps({
            "hl_proxy_url": "socks5://USER:PASS@dc.decodo.com",
            "hl_proxy_ports": "10001-10003",
        }, indent=2))
        sys.exit(1)
    with open(secrets_path) as f:
        return json.load(f)


def parse_ports(ports_str: str) -> List[int]:
    """Parse '10001-10010' or '10001,10002,10005' into list of ints."""
    ports = []
    for part in ports_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ports.extend(range(int(start.strip()), int(end.strip()) + 1))
        else:
            ports.append(int(part))
    return ports


# ─── IP verification ─────────────────────────────────────────────────────────

async def verify_proxy_ip(proxy_url: str, port: int) -> Optional[str]:
    """Verify the external IP of a proxy port via python-socks + raw HTTPS to httpbin."""
    try:
        sock = await create_proxy_socket(proxy_url, port, "httpbin.org", 443)

        ssl_ctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(
            sock=sock,
            ssl=ssl_ctx,
            server_hostname="httpbin.org",
        )

        request = (
            "GET /ip HTTP/1.1\r\n"
            "Host: httpbin.org\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(request.encode())
        await writer.drain()

        response = await asyncio.wait_for(reader.read(4096), timeout=10)
        writer.close()

        body = response.decode(errors="replace")
        json_start = body.find("{")
        json_end = body.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(body[json_start:json_end])
            return data.get("origin", "unknown")
        return "unknown (no JSON)"

    except Exception as e:
        return f"ERROR: {e}"


# ─── Wallet discovery ────────────────────────────────────────────────────────

def discover_wallets_rest(target_count: int) -> List[str]:
    """Discover active wallets via HL REST API (vault leaders)."""
    print(f"\n{C_BOLD}[DISCOVERY]{C_RESET} Finding active wallets via HL REST API (vaults)...")
    wallets: Set[str] = set()
    try:
        resp = requests.post(HL_INFO_URL, json={"type": "vaultSummaries"}, timeout=15)
        resp.raise_for_status()
        vaults = resp.json()
        if isinstance(vaults, list):
            for vault in vaults:
                for key in ("leader", "vaultAddress"):
                    addr = vault.get(key)
                    if addr and addr.startswith("0x") and len(addr) == 42:
                        wallets.add(addr)
                if len(wallets) >= target_count:
                    break
            print(f"  Found {len(wallets)} addresses from {len(vaults)} vaults")
    except Exception as e:
        print(f"  {C_YELLOW}Warning: Vault query failed: {e}{C_RESET}")

    if len(wallets) < target_count:
        print(f"  REST found {len(wallets)}/{target_count}, will supplement with WS...")
    return list(wallets)[:target_count]


async def discover_wallets_ws(
    target_count: int,
    timeout_s: float,
    proxy_url: Optional[str] = None,
    proxy_port: Optional[int] = None,
) -> List[str]:
    """Discover active wallets by subscribing to 'trades' on top coins via HL WS."""
    label = f"via proxy port {proxy_port}" if proxy_url else "direct"
    print(f"\n{C_BOLD}[DISCOVERY]{C_RESET} Listening to trades ({label}) for {timeout_s}s to find {target_count} wallets...")

    wallets: Set[str] = set()
    coins = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "HYPE", "SUI", "LINK", "AVAX"]
    trade_count = 0

    ws = None
    try:
        if proxy_url and proxy_port:
            ws_ctx = await connect_ws_proxied(proxy_url, proxy_port)
        else:
            ws_ctx = await connect_ws_direct()

        ws = await ws_ctx
        for coin in coins:
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": coin},
            }))

        deadline = time.time() + timeout_s
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if len(wallets) >= target_count:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("channel") == "trades":
                for trade in msg.get("data", []):
                    for user in trade.get("users", []):
                        if user and user.startswith("0x") and len(user) == 42:
                            if user not in wallets:
                                wallets.add(user)
                                print(f"  Found wallet: {user}")
                            trade_count += 1
                            if len(wallets) >= target_count:
                                break
                    if len(wallets) >= target_count:
                        break

    except Exception as e:
        print(f"  {C_RED}Discovery WS error: {e}{C_RESET}")
    finally:
        if ws is not None:
            try:
                # Proxy-tunneled sockets can occasionally stall on close handshake.
                await asyncio.wait_for(ws.close(), timeout=2.0)
            except Exception:
                transport = getattr(ws, "transport", None)
                if transport is not None:
                    transport.abort()

    result = list(wallets)[:target_count]
    print(f"  Discovered {len(result)} unique wallets from {trade_count} trade events")
    return result


# ─── HyperliquidTracker test doubles ─────────────────────────────────────────

class _MockEntityClient:
    """Minimal EntityClient surface needed by HyperliquidTracker."""

    def __init__(self, addresses: List[str], account_size: float = 10_000.0):
        self._addresses = list(addresses)
        self._account_size = account_size
        self._addr_to_hotkey = {a: f"hl_test_{i:03d}" for i, a in enumerate(self._addresses)}

    def get_all_active_hl_subaccounts(self):
        return [(a, {"source": "test"}) for a in self._addresses]

    def get_synthetic_hotkey_for_hl_address(self, hl_address: str):
        return self._addr_to_hotkey.get(hl_address)

    def get_subaccount_info_for_synthetic(self, synthetic_hotkey: str):
        if synthetic_hotkey is None:
            return None
        return {"account_size": self._account_size}

    def validate_hotkey_for_orders(self, synthetic_hotkey: str):
        if synthetic_hotkey is None:
            return {"is_valid": False, "error_message": "missing hotkey"}
        return {"is_valid": True, "error_message": ""}


class _MockEliminationClient:
    def get_elimination_local_cache(self, synthetic_hotkey: str):
        return None


class _MockPriceFetcherClient:
    def is_market_open(self, trade_pair, now_ms: int) -> bool:
        return True


class _MockUUIDTracker:
    def __init__(self):
        self.tracked: Set[str] = set()

    def add(self, order_uuid: str):
        self.tracked.add(order_uuid)


def _patch_tracker_dependencies(secrets: dict):
    """
    Patch secrets + OrderProcessor for standalone script execution.
    Returns a restore() callable.
    """
    from vali_objects.utils.limit_order.order_processor import OrderProcessor
    from vali_objects.utils.vali_utils import ValiUtils

    orig_get_secrets = ValiUtils.get_secrets
    orig_process_order = OrderProcessor.process_order

    ValiUtils.get_secrets = staticmethod(lambda: secrets)
    OrderProcessor.process_order = staticmethod(
        lambda **kwargs: SimpleNamespace(should_track_uuid=False)
    )

    def restore():
        ValiUtils.get_secrets = orig_get_secrets
        OrderProcessor.process_order = orig_process_order

    return restore


# ─── Main test orchestrator ──────────────────────────────────────────────────

async def run_test(
    wallets_to_find: int,
    watch_seconds: float,
    discovery_seconds: float,
):
    # 1. Load secrets
    secrets = load_secrets()
    proxy_base = secrets.get("hl_proxy_url")
    ports_str = secrets.get("hl_proxy_ports")

    if not proxy_base or not ports_str:
        print(f"{C_RED}ERROR: hl_proxy_url and hl_proxy_ports must be set in secrets.json{C_RESET}")
        sys.exit(1)

    proxy_base = proxy_base.rstrip("/")
    ports = parse_ports(ports_str)
    print(f"{C_BOLD}[CONFIG]{C_RESET} Proxy base: {proxy_base}")
    print(f"{C_BOLD}[CONFIG]{C_RESET} Ports: {ports}")
    print(f"{C_BOLD}[CONFIG]{C_RESET} Max addresses per shard: {MAX_ADDRS_PER_SHARD}")
    max_trackable = len(ports) * MAX_ADDRS_PER_SHARD
    print(f"{C_BOLD}[CONFIG]{C_RESET} Total trackable addresses: {max_trackable}")

    # 2. Verify proxy IPs are distinct
    print(f"\n{C_BOLD}[IP CHECK]{C_RESET} Verifying proxy IPs (this may take a moment)...")
    ip_map: Dict[int, str] = {}
    for port in ports:
        ip = await verify_proxy_ip(proxy_base, port)
        ip_map[port] = ip
        ok = ip and not ip.startswith("ERROR")
        status = f"{C_GREEN}OK{C_RESET}" if ok else f"{C_RED}FAIL{C_RESET}"
        print(f"  Port {port} -> {ip} [{status}]")

    valid_ips = {p: ip for p, ip in ip_map.items() if ip and not ip.startswith("ERROR")}
    unique_ips = set(valid_ips.values())
    if len(unique_ips) < len(valid_ips):
        print(f"  {C_YELLOW}WARNING: Some ports share the same IP! "
              f"({len(unique_ips)} unique for {len(valid_ips)} ports){C_RESET}")
    elif valid_ips:
        print(f"  {C_GREEN}All {len(unique_ips)} ports have distinct IPs{C_RESET}")

    working_ports = [p for p in ports if p in valid_ips]
    if not working_ports:
        print(f"{C_RED}ERROR: No working proxy ports found!{C_RESET}")
        sys.exit(1)
    print(f"  Using {len(working_ports)} working ports: {working_ports}")

    # 3. Discover wallets (REST first, then supplement with WS trades)
    wallets = discover_wallets_rest(wallets_to_find)

    if len(wallets) < wallets_to_find:
        remaining = wallets_to_find - len(wallets)
        ws_wallets = await discover_wallets_ws(
            remaining, discovery_seconds,
            proxy_url=proxy_base, proxy_port=working_ports[0],
        )
        if not ws_wallets:
            print(f"  {C_YELLOW}Proxy WS discovery failed, trying direct...{C_RESET}")
            ws_wallets = await discover_wallets_ws(remaining, discovery_seconds)
        existing = set(wallets)
        for w in ws_wallets:
            if w not in existing:
                wallets.append(w)
                existing.add(w)
            if len(wallets) >= wallets_to_find:
                break

    if not wallets:
        print(f"{C_RED}ERROR: Could not discover any wallets. Try increasing --discovery-seconds.{C_RESET}")
        sys.exit(1)

    print(f"\n{C_BOLD}[WALLETS]{C_RESET} Selected {len(wallets)} wallets to track:")
    for i, w in enumerate(wallets):
        print(f"  {i+1:3d}. {w}")

    # 4. Run the real HyperliquidTracker class with mocked dependencies
    print(f"\n{C_BOLD}[TRACKER]{C_RESET} Starting HyperliquidTracker for {watch_seconds}s... (Ctrl+C to stop early)\n")

    from entity_management.hyperliquid_tracker import HyperliquidTracker

    restore_patches = _patch_tracker_dependencies(secrets)
    tracker = HyperliquidTracker(
        entity_client=_MockEntityClient(wallets),
        elimination_client=_MockEliminationClient(),
        price_fetcher_client=_MockPriceFetcherClient(),
        asset_selection_client=None,
        market_order_manager=None,
        limit_order_client=None,
        uuid_tracker=_MockUUIDTracker(),
    )

    start_time = time.time()
    tracker.start()
    try:
        while (time.time() - start_time) < watch_seconds:
            await asyncio.sleep(15)
            status = tracker.get_status()
            elapsed = time.time() - start_time
            print(
                f"\n  {C_DIM}--- Status @ {elapsed:.0f}s: "
                f"{status['total_connected']}/{len(status['shards'])} connected, "
                f"fills_processed={status['fills_processed']}, "
                f"subscribed={status['total_subscribed_addresses']} ---{C_RESET}\n"
            )
    finally:
        tracker.stop()
        restore_patches()

    # 5. Summary
    elapsed = time.time() - start_time
    status = tracker.get_status()
    shard_statuses = status.get("shards", [])
    print(f"\n{'='*70}")
    print(f"{C_BOLD}  TEST SUMMARY{C_RESET}")
    print(f"{'='*70}")
    print(f"  Duration:        {elapsed:.1f}s")
    print(f"  Wallets tracked: {len(wallets)}")
    print(f"  Shards used:     {len(shard_statuses)}")
    print()

    print(f"  {'Shard':<25} {'Port':<8} {'IP':<18} {'Healthy':<9} {'Connected':<10} {'Addrs':<7}")
    print(f"  {'-'*25} {'-'*8} {'-'*18} {'-'*9} {'-'*10} {'-'*7}")
    for shard in shard_statuses:
        port = shard.get("port")
        ip = valid_ips.get(port, "?")
        healthy = shard.get("healthy", False)
        connected = shard.get("connected", False)
        health = f"{C_GREEN}yes{C_RESET}" if healthy else f"{C_RED}NO{C_RESET}"
        conn = f"{C_GREEN}yes{C_RESET}" if connected else f"{C_YELLOW}no{C_RESET}"
        print(
            f"  {('SHARD-' + str(shard.get('shard_id', '?'))):<25} "
            f"{str(port):<8} {ip:<18} {health:<18} {conn:<19} {shard.get('address_count', 0):<7}"
        )

    total_fills = status.get("fills_processed", 0)
    total_unhealthy = sum(1 for s in shard_statuses if not s.get("healthy", False))
    print()
    print(f"  Total processed fills: {total_fills}")
    print(f"  Unhealthy shards:     {total_unhealthy}")
    print(f"{'='*70}")

    if total_fills > 0:
        print(f"\n  {C_GREEN}SUCCESS: HyperliquidTracker processed live fills across shards!{C_RESET}")
    else:
        print(f"\n  {C_YELLOW}NO FILLS: Tracker ran but processed no fills in this window.{C_RESET}")
        print(f"  This is normal if discovered wallets were inactive during the watch period.")


def main():
    parser = argparse.ArgumentParser(
        description="Test Hyperliquid WebSocket proxy sharding with Decodo SOCKS5 proxies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test: find 15 wallets, watch for 60s
  python runnable/test_hl_proxy_sharding.py --wallets 15 --watch-seconds 60

  # Full test: find 30 wallets across 3 proxy IPs, watch for 5 minutes
  python runnable/test_hl_proxy_sharding.py --wallets 30 --watch-seconds 300

  # Fast discovery + long watch
  python runnable/test_hl_proxy_sharding.py --wallets 20 --discovery-seconds 30 --watch-seconds 600
""",
    )
    parser.add_argument(
        "--wallets", type=int, default=15,
        help="Number of wallets to discover and track (default: 15)",
    )
    parser.add_argument(
        "--watch-seconds", type=float, default=120,
        help="How long to watch for fills in seconds (default: 120)",
    )
    parser.add_argument(
        "--discovery-seconds", type=float, default=20,
        help="How long to listen to allTrades for wallet discovery (default: 20)",
    )
    args = parser.parse_args()

    print(f"{C_BOLD}{'='*70}{C_RESET}")
    print(f"{C_BOLD}  Hyperliquid WebSocket Proxy Sharding Test{C_RESET}")
    print(f"{C_BOLD}{'='*70}{C_RESET}")
    print(f"  Wallets to find:    {args.wallets}")
    print(f"  Discovery window:   {args.discovery_seconds}s")
    print(f"  Watch window:       {args.watch_seconds}s")

    try:
        asyncio.run(run_test(
            wallets_to_find=args.wallets,
            watch_seconds=args.watch_seconds,
            discovery_seconds=args.discovery_seconds,
        ))
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}Interrupted by user{C_RESET}")


if __name__ == "__main__":
    main()
