import os
from enum import Enum
from multiprocessing import Manager, Pool
import bittensor as bt


class IPCMetagraph:
    """
    IPC-compatible metagraph with getter methods and synthetic DEVELOPMENT hotkey
    for development testing.

    The DEVELOPMENT hotkey is a synthetic hotkey that:
    - Always appears in get_hotkeys() at index 0
    - Always returns True for has_hotkey("DEVELOPMENT")
    - Bypasses elimination checks and re-registration checks
    - Is NOT stored in the actual IPC metagraph (no pollution)

    This allows developers to test order processing without registering to the network.
    """

    DEVELOPMENT_HOTKEY = "DEVELOPMENT"

    def __init__(self, ipc_metagraph_namespace):
        """
        Initialize wrapper around IPC metagraph namespace.

        Args:
            ipc_metagraph_namespace: The Manager.Namespace() returned by get_ipc_metagraph()
        """
        # Use object.__setattr__ to avoid recursion with our custom __setattr__
        object.__setattr__(self, '_ipc_metagraph', ipc_metagraph_namespace)

        bt.logging.info(
            f"Metagraph wrapper initialized - '{self.DEVELOPMENT_HOTKEY}' hotkey "
            f"will be available for development orders"
        )

    def get_hotkeys(self):
        """
        Get list of all hotkeys including synthetic DEVELOPMENT hotkey.

        Returns:
            list: Hotkeys with DEVELOPMENT prepended at index 0
        """
        ipc_hotkeys = self._ipc_metagraph.hotkeys

        # Check if DEVELOPMENT is already in the actual list (shouldn't happen)
        if self.DEVELOPMENT_HOTKEY in ipc_hotkeys:
            bt.logging.warning(f"DEVELOPMENT hotkey found in actual metagraph - this should not happen")
            return list(ipc_hotkeys)

        # Return new list with DEVELOPMENT prepended
        return [self.DEVELOPMENT_HOTKEY] + list(ipc_hotkeys)

    def has_hotkey(self, hotkey: str) -> bool:
        """
        Check if a hotkey exists in the metagraph.

        Args:
            hotkey: The hotkey to check

        Returns:
            bool: True if hotkey is DEVELOPMENT or exists in actual metagraph
        """
        if hotkey == self.DEVELOPMENT_HOTKEY:
            return True
        return hotkey in self._ipc_metagraph.hotkeys

    def is_development_hotkey(self, hotkey: str) -> bool:
        """
        Check if a hotkey is the synthetic DEVELOPMENT hotkey.

        Args:
            hotkey: The hotkey to check

        Returns:
            bool: True if hotkey is DEVELOPMENT
        """
        return hotkey == self.DEVELOPMENT_HOTKEY

    def get_neurons(self):
        """
        Get list of neurons from metagraph.

        Note: DEVELOPMENT hotkey has no neuron entry - callers must handle this.

        Returns:
            list: Neuron objects from IPC metagraph
        """
        return self._ipc_metagraph.neurons

    def get_uids(self):
        """
        Get list of UIDs from metagraph.

        Returns:
            list: UIDs from IPC metagraph
        """
        return self._ipc_metagraph.uids

    def get_axons(self):
        """
        Get list of axons from metagraph.

        Returns:
            list: Axons from IPC metagraph
        """
        return self._ipc_metagraph.axons

    def get_emission(self):
        """
        Get emission data from metagraph.

        Returns:
            list: Emission values from IPC metagraph
        """
        return self._ipc_metagraph.emission

    # Direct property access for backwards compatibility during migration
    # These will be deprecated once all code uses getters

    @property
    def hotkeys(self):
        """Legacy property access - prefer get_hotkeys()."""
        return self.get_hotkeys()

    @hotkeys.setter
    def hotkeys(self, value):
        """Allow setting hotkeys (used by MetagraphUpdater sync)."""
        self._ipc_metagraph.hotkeys = value

    @property
    def neurons(self):
        """Direct access to neurons (used by MetagraphUpdater sync)."""
        return self._ipc_metagraph.neurons

    @neurons.setter
    def neurons(self, value):
        """Allow setting neurons (used by MetagraphUpdater sync)."""
        self._ipc_metagraph.neurons = value

    @property
    def uids(self):
        """Direct access to uids (used by MetagraphUpdater sync)."""
        return self._ipc_metagraph.uids

    @uids.setter
    def uids(self, value):
        """Allow setting uids (used by MetagraphUpdater sync)."""
        self._ipc_metagraph.uids = value

    @property
    def axons(self):
        """Direct access to axons (used by MetagraphUpdater sync)."""
        return self._ipc_metagraph.axons

    @axons.setter
    def axons(self, value):
        """Allow setting axons (used by MetagraphUpdater sync)."""
        self._ipc_metagraph.axons = value

    @property
    def block_at_registration(self):
        """Direct access to block_at_registration."""
        return self._ipc_metagraph.block_at_registration

    @block_at_registration.setter
    def block_at_registration(self, value):
        """Allow setting block_at_registration."""
        self._ipc_metagraph.block_at_registration = value

    @property
    def emission(self):
        """Direct access to emission."""
        return self._ipc_metagraph.emission

    @emission.setter
    def emission(self, value):
        """Allow setting emission."""
        self._ipc_metagraph.emission = value

    def __getattr__(self, name):
        """
        Proxy all other attribute accesses to the underlying IPC metagraph.
        This handles tao_reserve_rao, alpha_reserve_rao, tao_to_usd_rate, etc.
        """
        # Avoid infinite recursion
        if name == '_ipc_metagraph':
            return object.__getattribute__(self, name)

        # Proxy to underlying IPC metagraph
        return getattr(self._ipc_metagraph, name)

    def __setattr__(self, name, value):
        """
        Proxy all attribute assignments to the underlying IPC metagraph.
        This allows in-place updates to work correctly.
        """
        # Special handling for wrapper's own attributes
        if name == '_ipc_metagraph':
            object.__setattr__(self, name, value)
        else:
            # Proxy to underlying IPC metagraph
            setattr(self._ipc_metagraph, name, value)

    def __repr__(self):
        """String representation showing metagraph status."""
        try:
            hotkeys_count = len(self._ipc_metagraph.hotkeys)
            return f"<IPCMetagraph: {hotkeys_count} hotkeys (+1 DEVELOPMENT)>"
        except:
            return f"<IPCMetagraph: uninitialized>"


def get_ipc_metagraph(manager: Manager):
    """
    Create IPC-compatible metagraph with getter methods and synthetic DEVELOPMENT hotkey
    for development testing.

    Args:
        manager: The multiprocessing.Manager() instance

    Returns:
        IPCMetagraph: IPC-compatible metagraph with getter methods and DEVELOPMENT hotkey support
    """
    metagraph = manager.Namespace()
    metagraph.neurons = manager.list()
    metagraph.hotkeys = manager.list()
    metagraph.uids = manager.list()
    metagraph.block_at_registration = manager.list()
    # Substrate reserve balances (refreshed periodically by MetagraphUpdater)
    # Use manager.Value() for thread-safe float synchronization with internal locking
    metagraph.tao_reserve_rao = manager.Value('d', 0.0)  # 'd' = ctypes double (float64)
    metagraph.alpha_reserve_rao = manager.Value('d', 0.0)
    metagraph.emission = manager.list()  # TAO emission per tempo for each UID

    # Return IPCMetagraph with getter methods and DEVELOPMENT hotkey
    return IPCMetagraph(metagraph)

def managerize_objects(cls, manager, obj_dict) -> None:
    """
    Converts objects into manager-compatible shared objects and
    sets them as attributes of the validator object.

    Args:
        manager: The multiprocessing.Manager() instance.
        obj_dict: A dictionary of objects to managerize {name: object}.
    """

    def simple_managerize(obj):
        # Handle the special case for the 'metagraph' object
        if name == "metagraph":
            temp = manager.Namespace()
            temp.neurons = manager.list()
            temp.hotkeys = manager.list()
            temp.uids = manager.list()
            return temp

        # Managerize dictionaries
        elif isinstance(obj, dict):
            managed_dict = manager.dict()
            return managed_dict

        # Managerize lists
        elif isinstance(obj, list):
            managed_list = manager.list()
            return managed_list
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")

    # Managerize each object, with special handling for 'metagraph'
    for name, obj in obj_dict.items():
        setattr(cls, name, simple_managerize(obj))


class ParallelizationMode(Enum):
    SERIAL = 0
    PYSPARK = 1
    MULTIPROCESSING = 2

def get_multiprocessing_pool(parallel_mode: ParallelizationMode, num_processes: int = 0):
    print(f"parallel_mode: {parallel_mode} ({type(parallel_mode)})")
    pool = None
    if parallel_mode == ParallelizationMode.MULTIPROCESSING:
        print("Creating multiprocessing pool...")
        pool = Pool(num_processes) if num_processes else Pool()
        print(f"Pool created: {pool}")
    else:
        print("Not using multiprocessing mode.")
    return pool
def get_spark_session(parallel_mode: ParallelizationMode):
    if parallel_mode == ParallelizationMode.PYSPARK:
        # Check if running in Databricks
        is_databricks = 'DATABRICKS_RUNTIME_VERSION' in os.environ
        # Initialize Spark
        if is_databricks:
            # In Databricks, 'spark' is already available in the global namespace
            print("Running in Databricks environment, using existing spark session")
            should_close = False

        else:
            # Create a new Spark session if not in Databricks
            from pyspark.sql import SparkSession

            print("getOrCreate Spark session")
            spark = SparkSession.builder \
                .appName("PerfLedgerManager") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "6g") \
                .config("spark.executor.cores", "4") \
                .config("spark.driver.maxResultSize", "2g") \
                .getOrCreate()
            should_close = True

    else:
        spark = None
        should_close = False

    return spark, should_close
