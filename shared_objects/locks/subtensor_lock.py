# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import threading

# Global lock to synchronize subtensor operations (metagraph updates and weight setting)
# This prevents WebSocket concurrency errors when multiple threads try to use the same connection
_subtensor_operation_lock = threading.RLock()

def get_subtensor_lock():
    """
    Get the global subtensor operation lock.
    
    This lock should be used to synchronize operations that use the same WebSocket connection
    to the Bittensor network, such as:
    - Metagraph updates
    - Weight setting
    - Other subtensor operations
    
    Usage:
        from shared_objects.subtensor_lock import get_subtensor_lock
        
        with get_subtensor_lock():
            # Perform subtensor operation
            metagraph_clone = subtensor.metagraph(netuid)
            # or
            subtensor.set_weights(...)
    """
    return _subtensor_operation_lock