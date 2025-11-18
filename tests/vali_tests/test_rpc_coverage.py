# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Test to ensure all public server methods are exposed via RPC.

This test prevents issues where server methods are called directly in production
but aren't exposed via RPC, causing AttributeError at runtime.
"""

import inspect
import unittest
from vali_objects.utils.challengeperiod_manager_server import ChallengePeriodManagerServer
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager_server import EliminationManagerServer
from vali_objects.utils.elimination_manager import EliminationManager


class TestRPCCoverage(unittest.TestCase):
    """
    Test that all public server methods are properly exposed via RPC.

    This catches issues like:
    - Server method `foo()` exists but no `foo_rpc()` exposed
    - Client calls `server.foo()` instead of `_server_proxy.foo_rpc()`
    """

    # Methods that are intentionally not exposed via RPC (e.g., static helpers, internal methods)
    CHALLENGEPERIOD_ALLOWED_MISSING = {
        # Static methods don't need RPC exposure
        'parse_checkpoint_dict',
        'screen_minimum_interaction',
        'screen_minimum_ledger',
        'screen_minimum_positions',
        'is_recently_re_registered',

        # Internal lifecycle methods (inherited from CacheController)
        'refresh_allowed',
        'set_last_update_time',
        'get_last_update_time_ms',
        'get_last_modified_time_miner_directory',
        'init_cache_files',
        'is_drawdown_beyond_mdd',
        'get_directory_names',

        # Methods that run in daemon loop (not meant to be called via RPC)
        'run_update_loop',  # Daemon entry point
        'add_all_miners_to_success',  # Backtesting only

        # Internal helper methods (should probably be made private)
        'evaluate_promotions',
        'remove_eliminated',
        'update_plagiarism_miners',
        'prepare_plagiarism_elimination_miners',
        'meets_time_criteria',
        'sync_challenge_period_data',
        'refresh',
        'inspect',
        'get_hotkeys_by_bucket',
        'generate_elimination_row',
        'calculate_drawdown',
        'iter_active_miners',
    }

    ELIMINATION_ALLOWED_MISSING = {
        # Static methods don't need RPC exposure

        # Internal lifecycle methods (inherited from CacheController)
        'refresh_allowed',
        'set_last_update_time',
        'get_last_update_time_ms',
        'get_last_modified_time_miner_directory',
        'init_cache_files',
        'is_drawdown_beyond_mdd',
        'get_directory_names',

        # Methods that run in daemon loop (not meant to be called via RPC)
        'run_update_loop',  # Daemon entry point

        # Internal helper methods
        'handle_zombies',
        'calculate_drawdown',
        'handle_eliminated_miner',
        'generate_elimination_row',
        'handle_challenge_period_eliminations',
        'add_manual_flat_order',
        'delete_eliminations',  # Composite method that loops over remove_elimination
    }

    CLIENT_COMPOSITE_METHODS = {
        # Client methods that compose multiple RPC calls (don't directly call _server_proxy)
        'iter_active_miners',  # Composes get_testing_miners, get_success_miners, etc.
    }

    def _get_public_methods(self, cls):
        """
        Get all public methods of a class (not starting with _).

        Returns:
            set: Set of method names
        """
        methods = set()
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip private methods, special methods, and properties
            if not name.startswith('_'):
                methods.add(name)
        return methods

    def _get_rpc_methods(self, cls):
        """
        Get all RPC methods (ending with _rpc).

        Returns:
            dict: Mapping from base method name to RPC method name
                  e.g., {'get_miner_bucket': 'get_miner_bucket_rpc'}
        """
        rpc_methods = {}
        for name in dir(cls):
            if name.endswith('_rpc'):
                base_name = name[:-4]  # Remove '_rpc' suffix
                rpc_methods[base_name] = name
        return rpc_methods

    def test_challengeperiod_server_methods_have_rpc_exposure(self):
        """
        Verify all public ChallengePeriodManagerServer methods are exposed via RPC.
        """
        public_methods = self._get_public_methods(ChallengePeriodManagerServer)
        rpc_methods = self._get_rpc_methods(ChallengePeriodManagerServer)

        missing_rpc = []
        for method_name in public_methods:
            # Skip methods that are already RPC methods (end with _rpc)
            if method_name.endswith('_rpc'):
                continue
            if method_name in self.CHALLENGEPERIOD_ALLOWED_MISSING:
                continue
            if method_name not in rpc_methods:
                missing_rpc.append(method_name)

        if missing_rpc:
            error_msg = (
                f"\n\nChallengePeriodManagerServer has public methods without RPC exposure:\n"
                f"{missing_rpc}\n\n"
                f"For each method, either:\n"
                f"1. Add '{{}}_rpc()' method to server that calls '{{}}')\n"
                f"2. Make method private (rename to '_{{}}')\n"
                f"3. Add to CHALLENGEPERIOD_ALLOWED_MISSING if intentionally not exposed\n"
            )
            self.fail(error_msg)

    def test_elimination_server_methods_have_rpc_exposure(self):
        """
        Verify all public EliminationManagerServer methods are exposed via RPC.
        """
        public_methods = self._get_public_methods(EliminationManagerServer)
        rpc_methods = self._get_rpc_methods(EliminationManagerServer)

        missing_rpc = []
        for method_name in public_methods:
            # Skip methods that are already RPC methods (end with _rpc)
            if method_name.endswith('_rpc'):
                continue
            if method_name in self.ELIMINATION_ALLOWED_MISSING:
                continue
            if method_name not in rpc_methods:
                missing_rpc.append(method_name)

        if missing_rpc:
            error_msg = (
                f"\n\nEliminationManagerServer has public methods without RPC exposure:\n"
                f"{missing_rpc}\n\n"
                f"For each method, either:\n"
                f"1. Add '{{}}_rpc()' method to server that calls '{{}}')\n"
                f"2. Make method private (rename to '_{{}}')\n"
                f"3. Add to ELIMINATION_ALLOWED_MISSING if intentionally not exposed\n"
            )
            self.fail(error_msg)

    def test_challengeperiod_client_methods_use_rpc(self):
        """
        Verify ChallengePeriodManager client methods properly use RPC.

        Checks that client methods either:
        - Call _server_proxy.<method>_rpc()
        - Are static methods
        - Are properties
        """
        # Get all public methods from client
        client_methods = self._get_public_methods(ChallengePeriodManager)

        # Get source code for client class
        client_source = inspect.getsource(ChallengePeriodManager)

        issues = []
        for method_name in client_methods:
            # Skip special cases
            if method_name in {'parse_checkpoint_dict', 'screen_minimum_interaction'}:
                # These are static methods
                continue

            # Skip inherited CacheController methods
            if method_name in {'refresh_allowed', 'set_last_update_time', 'get_last_update_time_ms',
                             'init_cache_files', 'is_drawdown_beyond_mdd',
                             'get_last_modified_time_miner_directory'}:
                continue

            # Skip composite methods
            if method_name in self.CLIENT_COMPOSITE_METHODS:
                continue

            # Get method object
            method = getattr(ChallengePeriodManager, method_name)

            # Skip properties and static methods
            if isinstance(inspect.getattr_static(ChallengePeriodManager, method_name), (property, staticmethod)):
                continue

            # Check if method uses RPC
            try:
                method_source = inspect.getsource(method)

                # Method should either:
                # 1. Call _server_proxy.*_rpc()
                # 2. Be a simple getter/setter for properties
                if '_server_proxy' not in method_source and 'return self._' not in method_source:
                    issues.append(f"{method_name}() doesn't appear to use RPC (_server_proxy)")
            except (OSError, TypeError):
                # Can't get source for some methods (built-ins, etc.) - skip them
                pass

        if issues:
            error_msg = (
                f"\n\nChallengePeriodManager client methods not using RPC:\n"
                f"{chr(10).join(issues)}\n\n"
                f"Client methods should call self._server_proxy.<method>_rpc()\n"
            )
            self.fail(error_msg)

    def test_elimination_client_methods_use_rpc(self):
        """
        Verify EliminationManager client methods properly use RPC.
        """
        # Get all public methods from client
        client_methods = self._get_public_methods(EliminationManager)

        issues = []
        for method_name in client_methods:
            # Skip inherited CacheController methods
            if method_name in {'refresh_allowed', 'set_last_update_time', 'get_last_update_time_ms',
                              'get_directory_names', 'is_drawdown_beyond_mdd',
                              'get_last_modified_time_miner_directory', 'init_cache_files'}:
                continue

            # Skip internal helper methods and performance-optimized cache methods
            if method_name in {'calculate_drawdown', 'generate_elimination_row', 'get_eliminations_lock',
                              'get_cached_elimination_data'}:  # Intentionally uses local cache (no RPC) for performance
                continue

            # Get method object
            method = getattr(EliminationManager, method_name)

            # Skip properties
            if isinstance(inspect.getattr_static(EliminationManager, method_name), property):
                continue

            # Check if method uses RPC
            try:
                method_source = inspect.getsource(method)

                # Method should either:
                # 1. Call _server_proxy.*_rpc()
                # 2. Be a simple getter/setter for properties
                # 3. Be delete_eliminations which loops and calls RPC methods
                if method_name == 'delete_eliminations':
                    # This method is allowed to not call _server_proxy directly
                    # because it loops over individual remove_elimination() calls
                    continue

                if '_server_proxy' not in method_source and 'return self._' not in method_source:
                    issues.append(f"{method_name}() doesn't appear to use RPC (_server_proxy)")
            except (OSError, TypeError):
                # Can't get source for some methods (built-ins, etc.) - skip them
                pass

        if issues:
            error_msg = (
                f"\n\nEliminationManager client methods not using RPC:\n"
                f"{chr(10).join(issues)}\n\n"
                f"Client methods should call self._server_proxy.<method>_rpc()\n"
            )
            self.fail(error_msg)


if __name__ == '__main__':
    unittest.main()
