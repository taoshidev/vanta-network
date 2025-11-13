from typing import List

from bittensor import Balance


class MockAxonInfo:
    ip: str

    def __init__(self, ip: str):
        self.ip = ip

class MockNeuron:
    axon_info: MockAxonInfo
    stake: Balance

    def __init__(self, axon_info: MockAxonInfo, stake: Balance):
        self.axon_info = axon_info
        self.stake = stake


class MockMetagraph():
    neurons: List[MockNeuron]
    hotkeys: List[str]
    uids: List[int]
    block_at_registration: List[int]

    def __init__(self, hotkeys, neurons = None):
        self.hotkeys = hotkeys
        self.neurons = neurons
        self.uids = []
        self.block_at_registration = []
        self.axons = []
        self.emission = []

    def __getstate__(self):
        """Support pickling for multiprocessing."""
        return {
            'hotkeys': self.hotkeys,
            'neurons': self.neurons,
            'uids': self.uids,
            'block_at_registration': self.block_at_registration,
            'axons': self.axons,
            'emission': self.emission,
        }

    def __setstate__(self, state):
        """Support unpickling for multiprocessing."""
        self.hotkeys = state['hotkeys']
        self.neurons = state['neurons']
        self.uids = state['uids']
        self.block_at_registration = state['block_at_registration']
        self.axons = state['axons']
        self.emission = state['emission']

    def get_hotkeys(self) -> List[str]:
        """Get list of all hotkeys."""
        return self.hotkeys

    def get_neurons(self) -> List[MockNeuron]:
        """Get list of all neurons."""
        return self.neurons

    def get_uids(self) -> List[int]:
        """Get list of all UIDs."""
        return self.uids

    def get_axons(self):
        """Get list of all axons."""
        return self.axons

    def get_emission(self):
        """Get emission values."""
        return self.emission

    def get_block_at_registration(self):
        """Get block at registration values."""
        return self.block_at_registration

    def has_hotkey(self, hotkey: str) -> bool:
        """Check if hotkey exists in metagraph."""
        return hotkey in self.hotkeys

    def is_development_hotkey(self, hotkey: str) -> bool:
        """Check if hotkey is the synthetic DEVELOPMENT hotkey."""
        return hotkey == "DEVELOPMENT"