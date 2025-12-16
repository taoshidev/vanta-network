import json
import os
import threading
import time
import traceback
from multiprocessing import current_process


class APIKeyMixin:
    """
    Mixin class that handles API key loading and validation with tiered permissions.

    API Keys JSON File Format

    This file explains the structure of the api_keys.json file used by the PTN Data API Server.
    The file supports two formats for backward compatibility:

    1. Modern Format (Recommended)
    --------------------------
    Each user entry is a dictionary with the following structure:
    {
      "user_id": {
        "key": "the_api_key_string",
        "tier": integer_tier_value
      },
      ...
    }

    Where tier values represent access levels:
    - 0: Basic access (24-hour lagged data, no real-time access)
    - 30: Standard access (30% real-time data)
    - 50: Enhanced access (50% real-time data)
    - 100: Premium access (100% data freshness + WebSocket support)

    2. Legacy Format (Supported but not recommended)
    -----------------------------------------
    Each user entry is a simple key-value pair:
    {
      "user_id": "the_api_key_string"
    }

    In this format, all keys are assigned a default tier of 100 (full access).

    Example Usage
    -------------
    To create a new API key file:

    1. Create a JSON file at path: vanta_api/api_keys.json
    2. Add user entries following either format above
    3. The server automatically loads and refreshes this file (every 15 seconds by default)
    4. You can add/remove/modify keys without restarting the server

    Security Notes
    -------------
    - Keep this file secure with appropriate permissions
    - In production, consider storing API keys in a database or secret management system
    - API keys should be treated as secrets and not committed to version control


    Example structure:

    {
      "user1": {
        "key": "diQDNkoB3urHC9yOFo7iZOsTo09S?2hm9u",
        "tier": 100
      },
      "user2": {
        "key": "tR5Bm3pLk9Xz7J2qV6sW0yA8dC4fG1hE",
        "tier": 50
      },
      "user3": {
        "key": "eK5hP9rT2xL7mN1bV6cZ3jQ8sW4dF0gA",
        "tier": 30
      },
      "trial_user": {
        "key": "aB9cD3eF7gH1jK5lM9nP2qR6sT0vW4xY",
        "tier": 0
      },
      "legacy_user": "oL5pM2nB9vC3xZ1aS7dF0gH4jK6qR8tW"
    }
    """


    def __init__(self, api_keys_file, refresh_interval=15):
        """Initialize API key handling functionality.

        Args:
            api_keys_file: Path to the JSON file containing API keys
            refresh_interval: How often to check for API key changes (seconds)
        """
        self.api_keys_file = api_keys_file
        self.refresh_interval = refresh_interval
        self.api_keys_data = {}  # Store full key data including tiers
        self.accessible_api_keys = []  # Keep for backwards compatibility
        self.last_modified_time = 0  # Track last modification time
        self.api_key_refresh_thread = None
        self.api_key_to_alias = {}  # api_key → user_id

        # Load API keys immediately
        try:
            self.load_api_keys()
            print(
                f"[{current_process().name}] Loaded {len(self.api_keys_data)} API keys from {self.api_keys_file}")
        except Exception as e:
            print(f"[{current_process().name}] Error loading API keys initially: {e}")
            print(traceback.format_exc())

    def create_default_api_keys_file(self, api_keys_file):
        """Create a default API keys file if it doesn't exist.

        Args:
            api_keys_file: Path to the API keys file. illustrates the expected structure of the api keys file.
        """
        if os.path.exists(api_keys_file):
            return

        default_keys = {
            "test_user": {
                "key": "free_test_key",
                "tier": 0
            },
            "premium_test_user": {
                "key": "premium_test_key",
                "tier": 100
            }
        }

        with open(api_keys_file, "w") as f:
            json.dump(default_keys, f, indent=2)

        print(f"Created default API keys file at {api_keys_file}")

    def load_api_keys(self):
        """Loads API keys from a file and logs changes only if the file has been modified."""
        if not os.path.exists(self.api_keys_file):
            raise FileNotFoundError(f"API keys file '{self.api_keys_file}' not found!")

        try:
            # Check file modification time
            current_mod_time = os.path.getmtime(self.api_keys_file)

            # Only reload if the file has been modified since last load
            if current_mod_time <= self.last_modified_time:
                # File hasn't changed, skip loading
                return

            # File has changed, update the last modified time
            self.last_modified_time = current_mod_time
            last_modified_str = time.ctime(current_mod_time)
            print(f"[{current_process().name}] API keys file last modified at: {last_modified_str}")

            with open(self.api_keys_file, "r") as f:
                new_keys = json.load(f)

            if not isinstance(new_keys, dict):
                raise ValueError("API keys file must contain a dict of key/values.")

            old_count = len(self.api_keys_data)

            # Process the new keys data format
            processed_keys = {}
            accessible_keys = []
            api_key_to_name = {}

            for user_id, key_data in new_keys.items():
                # Handle both legacy and new formats
                if isinstance(key_data, str):
                    # Legacy format: just an API key string
                    api_key = key_data
                    tier = 100  # Default tier for legacy keys
                    processed_keys[api_key] = {"tier": tier}
                    api_key_to_name[api_key] = user_id
                    accessible_keys.append(api_key)
                elif isinstance(key_data, dict):
                    # New format: dictionary with key and permissions
                    api_key = key_data.get("key", "")
                    tier = key_data.get("tier", 0)

                    if not api_key:
                        print(f"[{current_process().name}] Warning: Missing API key for user {user_id}")
                        continue

                    processed_keys[api_key] = {"tier": tier}
                    api_key_to_name[api_key] = user_id
                    accessible_keys.append(api_key)
                else:
                    print(f"[{current_process().name}] Warning: Invalid key data format for user {user_id}")

            new_count = len(processed_keys)
            # Only update and log if keys have actually changed
            if old_count != new_count or set(self.accessible_api_keys) != set(accessible_keys):
                print(f"[{current_process().name}] API key list size changed: {old_count} -> {new_count}")
                self.api_keys_data = processed_keys
                self.accessible_api_keys = accessible_keys
                self.api_key_to_alias = api_key_to_name
                print(
                    f"[{current_process().name}] Updated API keys data with {len(self.api_keys_data)} keys")

        except Exception as e:
            print(f"[{current_process().name}] Error loading API keys: {e}")
            print(traceback.format_exc())
            raise

    def refresh_api_keys(self):
        """Continuously refresh API keys every specified interval."""
        while True:
            try:
                self.load_api_keys()
            except Exception as e:
                print(f"[{current_process().name}] Failed to refresh API keys: {e}")
                print(traceback.format_exc())
            time.sleep(self.refresh_interval)

    def start_refresh_thread(self):
        """Start the API key refresh thread."""
        if self.api_key_refresh_thread is None or not self.api_key_refresh_thread.is_alive():
            self.api_key_refresh_thread = threading.Thread(target=self.refresh_api_keys, daemon=True)
            self.api_key_refresh_thread.start()
            print(f"[{current_process().name}] API key refresh thread started")

    def is_valid_api_key(self, api_key):
        """Checks if an API key is valid."""
        try:
            result = api_key in self.accessible_api_keys
            return result
        except Exception as e:
            print(f"[{current_process().name}] Error in is_valid_api_key: {e}")
            print(traceback.format_exc())
            return False

    def get_api_key_tier(self, api_key):
        """Get the tier level for an API key.

        Args:
            api_key: The API key to check

        Returns:
            int: The tier level (0, 30, 50, 100) or 0 if key is invalid
        """
        try:
            if not self.is_valid_api_key(api_key):
                return 0

            key_data = self.api_keys_data.get(api_key, {})
            return key_data.get("tier", 0)
        except Exception as e:
            print(f"[{current_process().name}] Error in get_api_key_tier: {e}")
            print(traceback.format_exc())
            return 0

    def can_access_tier(self, api_key, required_tier):
        """Check if an API key has sufficient tier access.

        Args:
            api_key: The API key to check
            required_tier: The minimum tier required for access

        Returns:
            bool: True if the API key has sufficient tier access
        """
        key_tier = self.get_api_key_tier(api_key)
        return key_tier >= required_tier