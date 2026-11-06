"""
HLFundingRateManager - Fetches, caches, and persists Hyperliquid funding rates.

Funding rates are settled hourly on HL. This manager maintains an in-memory cache
and persists to disk for crash recovery.
"""
import json
import os
import threading
import time
from bisect import bisect_left, bisect_right
from typing import Dict, List, Optional

import requests

from vali_objects.vali_config import ValiConfig
from shared_objects.log import logger


class HLFundingRateManager:
    """
    Manages HL funding rate data: fetch from API, cache in memory, persist to disk.

    Data structure: {coin: [{time_ms: int, rate: float}, ...]} sorted by time_ms.
    """

    PERSISTENCE_PATH = os.path.join(ValiConfig.BASE_DIR, "validation", "hl_funding_rates.json")

    def __init__(self, running_unit_tests: bool = False):
        self._lock = threading.Lock()
        # coin -> sorted list of {time_ms, rate}
        self._rates: Dict[str, List[dict]] = {}
        self._running_unit_tests = running_unit_tests
        if not running_unit_tests:
            self._load_from_disk()

    def _load_from_disk(self):
        """Load persisted funding rates from disk."""
        try:
            if os.path.exists(self.PERSISTENCE_PATH):
                with open(self.PERSISTENCE_PATH, "r") as f:
                    data = json.load(f)
                # Ensure sorted
                for coin, rates in data.items():
                    self._rates[coin] = sorted(rates, key=lambda r: r["time_ms"])
                logger.info(f"[HL_FUNDING] Loaded {sum(len(v) for v in self._rates.values())} funding rate records from disk")
        except Exception as e:
            logger.warning(f"[HL_FUNDING] Failed to load from disk: {e}")

    def _save_to_disk(self):
        """Persist funding rates to disk."""
        if self._running_unit_tests:
            return
        try:
            os.makedirs(os.path.dirname(self.PERSISTENCE_PATH), exist_ok=True)
            with open(self.PERSISTENCE_PATH, "w") as f:
                json.dump(self._rates, f)
        except Exception as e:
            logger.warning(f"[HL_FUNDING] Failed to save to disk: {e}")

    def fetch_and_store_rates(self, coins: List[str], start_ms: int, end_ms: int):
        """
        Fetch funding rates from HL API for the given coins and time range.
        Deduplicates and persists.
        """
        api_url = ValiConfig.hl_info_url()
        t0 = time.time()
        total_fetched = 0
        total_new = 0
        failed_coins = []
        logger.info(f"[HL_FUNDING] Fetching funding rates for {len(coins)} coins, "
                    f"start_ms={start_ms} end_ms={end_ms}")
        for coin in coins:
            try:
                resp = requests.post(api_url, json={
                    "type": "fundingHistory",
                    "coin": coin,
                    "startTime": start_ms,
                    "endTime": end_ms,
                }, timeout=15)
                raw_rates = resp.json()
                if not isinstance(raw_rates, list):
                    logger.warning(f"[HL_FUNDING] Unexpected fundingHistory response for {coin} "
                                   f"(status={resp.status_code}): {str(raw_rates)[:200]}")
                    failed_coins.append(coin)
                    continue

                new_records = []
                for r in raw_rates:
                    time_ms = int(r.get("time", 0))
                    rate = float(r.get("fundingRate", 0))
                    if time_ms > 0:
                        new_records.append({"time_ms": time_ms, "rate": rate})
                total_fetched += len(new_records)

                if new_records:
                    with self._lock:
                        existing = self._rates.get(coin, [])
                        existing_times = {r["time_ms"] for r in existing}
                        for rec in new_records:
                            if rec["time_ms"] not in existing_times:
                                existing.append(rec)
                                existing_times.add(rec["time_ms"])
                                total_new += 1
                        self._rates[coin] = sorted(existing, key=lambda r: r["time_ms"])

            except Exception as e:
                logger.warning(f"[HL_FUNDING] Failed to fetch rates for {coin}: {e}")
                failed_coins.append(coin)

        self._save_to_disk()
        logger.info(f"[HL_FUNDING] Fetched {total_fetched} rate records ({total_new} new) for "
                    f"{len(coins) - len(failed_coins)}/{len(coins)} coins in {time.time() - t0:.1f}s")
        if failed_coins:
            logger.warning(f"[HL_FUNDING] Failed to fetch rates for {len(failed_coins)} coin(s): {sorted(failed_coins)}")

    def get_rates_for_position(self, coin: str, open_ms: int, current_ms: int) -> Dict[int, float]:
        """
        Get funding rates between open_ms and current_ms for a coin.

        Returns: {settlement_time_ms: rate}
        """
        with self._lock:
            rates = self._rates.get(coin, [])

        if not rates:
            return {}

        # Binary search for range
        times = [r["time_ms"] for r in rates]
        left = bisect_left(times, open_ms)
        right = bisect_right(times, current_ms)

        return {rates[i]["time_ms"]: rates[i]["rate"] for i in range(left, right)}

    def get_rate_at_time(self, coin: str, time_ms: int) -> Optional[float]:
        """Get the funding rate at (or just before) a specific time."""
        with self._lock:
            rates = self._rates.get(coin, [])

        if not rates:
            return None

        times = [r["time_ms"] for r in rates]
        idx = bisect_right(times, time_ms) - 1
        if idx < 0:
            return None
        return rates[idx]["rate"]

    def clear_all(self):
        """Clear all cached rates (for testing)."""
        with self._lock:
            self._rates.clear()
