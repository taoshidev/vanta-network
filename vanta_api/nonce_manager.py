import time
import threading
from collections import defaultdict
from typing import Dict, Set
from time_util.time_util import TimeUtil

class NonceManager:
    def __init__(self, ttl_ms: int = 5 * 60 * 1000):  # 5 minute window
        self.ttl_ms = ttl_ms
        self.used_nonces: Dict[str, Set[str]] = defaultdict(set)  # address -> nonce_set
        self.nonce_timestamps: Dict[str, Dict[str, int]] = defaultdict(dict)  # address -> {nonce: timestamp}
        self.lock = threading.Lock()

    def is_valid_request(self, address: str, nonce: str, timestamp: int) -> tuple[bool, str]:
        """
        Validate timestamp and nonce for replay attack prevention.

        Returns:
            (is_valid, error_message)
        """
        current_time = TimeUtil.now_in_millis()

        # 1. Check timestamp freshness (prevent old replays)
        if current_time - timestamp > self.ttl_ms:
            return False, f"Request expired. Max age: {self.ttl_ms}ms"

        # 2. Check timestamp not too far in future (prevent future-dated attacks)
        if timestamp > current_time + 60 * 1000:  # 1 minute tolerance
            return False, "Request timestamp too far in future"

        with self.lock:
            # 3. Clean up expired nonces first
            self._cleanup_expired_nonces(address, current_time)

            # 4. Check if nonce already used
            if nonce in self.used_nonces[address]:
                return False, "Nonce already used"

            # 5. Mark nonce as used
            self.used_nonces[address].add(nonce)
            self.nonce_timestamps[address][nonce] = timestamp

        return True, ""

    def _cleanup_expired_nonces(self, address: str, current_time: int):
        """Remove expired nonces to prevent memory bloat."""
        expired_nonces = []

        for nonce, timestamp in self.nonce_timestamps[address].items():
            if current_time - timestamp > self.ttl_ms:
                expired_nonces.append(nonce)

        for nonce in expired_nonces:
            self.used_nonces[address].discard(nonce)
            del self.nonce_timestamps[address][nonce]
