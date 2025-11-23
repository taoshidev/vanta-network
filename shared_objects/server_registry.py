# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Server Registry for RPC Server Instance Tracking.

This module provides centralized tracking of all RPC server instances
for test cleanup and duplicate detection.
"""
import threading
from typing import Dict, List, TYPE_CHECKING
import bittensor as bt
from shared_objects.port_manager import PortManager
from vali_objects.vali_config import RPCConnectionMode

if TYPE_CHECKING:
    from shared_objects.rpc_server_base import RPCServerBase


class ServerRegistry:
    """
    Centralized registry for tracking all RPC server instances.

    Maintains registries by instance list, service name, and port to:
    - Prevent duplicate servers
    - Enable shutdown_all() for test cleanup
    - Track active servers for debugging

    This is a singleton-like class with class-level state.

    Example:
        # Register a server
        ServerRegistry.register(my_server)

        # Shutdown all servers (test cleanup)
        ServerRegistry.shutdown_all()

        # Force kill ports
        ServerRegistry.force_kill_all_rpc_ports()
    """

    # Class-level registry of all active server instances
    _active_instances: List['RPCServerBase'] = []
    _active_by_name: Dict[str, 'RPCServerBase'] = {}
    _active_by_port: Dict[int, 'RPCServerBase'] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def register(cls, instance: 'RPCServerBase') -> None:
        """
        Register a new server instance for tracking.

        Args:
            instance: The RPCServerBase instance to register

        Raises:
            RuntimeError: If a server with the same name or port is already registered
        """
        with cls._registry_lock:
            # Check for duplicate service name
            if instance.service_name in cls._active_by_name:
                existing = cls._active_by_name[instance.service_name]
                raise RuntimeError(
                    f"Duplicate RPC server: '{instance.service_name}' already registered "
                    f"(existing instance: {existing})"
                )

            # Check for duplicate port (only if in RPC mode - LOCAL mode doesn't use ports)
            if (instance.connection_mode == RPCConnectionMode.RPC and
                instance.port in cls._active_by_port):
                existing = cls._active_by_port[instance.port]
                raise RuntimeError(
                    f"Duplicate RPC port: port {instance.port} already in use by "
                    f"'{existing.service_name}' (new service: '{instance.service_name}')"
                )

            # Register the instance
            cls._active_instances.append(instance)
            cls._active_by_name[instance.service_name] = instance
            if instance.connection_mode == RPCConnectionMode.RPC:
                cls._active_by_port[instance.port] = instance

            bt.logging.debug(
                f"Registered {instance.service_name} "
                f"(total servers: {len(cls._active_instances)})"
            )

    @classmethod
    def unregister(cls, instance: 'RPCServerBase') -> None:
        """
        Unregister a server instance.

        Args:
            instance: The RPCServerBase instance to unregister
        """
        with cls._registry_lock:
            if instance in cls._active_instances:
                cls._active_instances.remove(instance)

            # Remove from name registry
            if instance.service_name in cls._active_by_name:
                if cls._active_by_name[instance.service_name] is instance:
                    del cls._active_by_name[instance.service_name]

            # Remove from port registry
            if instance.port in cls._active_by_port:
                if cls._active_by_port[instance.port] is instance:
                    del cls._active_by_port[instance.port]

            bt.logging.debug(
                f"Unregistered {instance.service_name} "
                f"(remaining servers: {len(cls._active_instances)})"
            )

    @classmethod
    def shutdown_all(cls, force_kill_ports: bool = True) -> None:
        """
        Shutdown all active server instances.

        Call this in test tearDown to ensure all servers are properly cleaned up
        before the next test starts. This prevents port conflicts between tests.

        Args:
            force_kill_ports: If True, force-kill any processes still using RPC ports
                            after graceful shutdown (default: True)

        Example:
            def tearDown(self):
                ServerRegistry.shutdown_all()
        """
        with cls._registry_lock:
            instances = list(cls._active_instances)
            ports_to_clean = [inst.port for inst in instances if hasattr(inst, 'port')]
            cls._active_instances.clear()
            cls._active_by_name.clear()
            cls._active_by_port.clear()

        for instance in instances:
            try:
                instance.shutdown()
            except Exception as e:
                bt.logging.trace(f"Error shutting down {instance.service_name}: {e}")

        # Force kill any remaining processes on these ports
        if force_kill_ports and ports_to_clean:
            cls.force_kill_ports(ports_to_clean)

        bt.logging.debug(f"Shutdown {len(instances)} RPC server instances")

    @classmethod
    def force_kill_ports(cls, ports: list) -> None:
        """
        Force-kill any processes using the specified ports.
        Delegates to PortManager.force_kill_ports().

        Args:
            ports: List of port numbers to force-kill
        """
        PortManager.force_kill_ports(ports)

    @classmethod
    def force_kill_all_rpc_ports(cls) -> None:
        """
        Force-kill any processes using any known RPC port.
        Delegates to PortManager.force_kill_all_rpc_ports().
        """
        PortManager.force_kill_all_rpc_ports()

    @classmethod
    def get_active_count(cls) -> int:
        """Get number of active registered servers."""
        with cls._registry_lock:
            return len(cls._active_instances)

    @classmethod
    def get_active_names(cls) -> List[str]:
        """Get list of active server names."""
        with cls._registry_lock:
            return list(cls._active_by_name.keys())

    @classmethod
    def get_active_ports(cls) -> List[int]:
        """Get list of active server ports."""
        with cls._registry_lock:
            return list(cls._active_by_port.keys())

    @classmethod
    def is_registered(cls, service_name: str) -> bool:
        """Check if a server with the given name is registered."""
        with cls._registry_lock:
            return service_name in cls._active_by_name

    @classmethod
    def get_by_name(cls, service_name: str) -> 'RPCServerBase':
        """
        Get server instance by service name.

        Args:
            service_name: Name of the service

        Returns:
            The registered RPCServerBase instance

        Raises:
            KeyError: If no server with that name is registered
        """
        with cls._registry_lock:
            return cls._active_by_name[service_name]
