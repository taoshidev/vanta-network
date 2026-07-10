# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
import json
import os
import threading
from datetime import datetime, timezone

from vali_objects.vali_config import ValiConfig


class AuditLogger:
    """
    Appends structured JSON Lines audit entries for tier-500 admin endpoints.
    Thread-safe; auto-creates the output directory on first use.
    """

    DEFAULT_PATH = os.path.join(ValiConfig.BASE_DIR, "vanta_api", "audit_log.jsonl")

    def __init__(self, path: str = DEFAULT_PATH):
        self._path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def log(self, entry: dict) -> None:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(entry, default=str)
        with self._lock:
            with open(self._path, "a") as f:
                f.write(line + "\n")
