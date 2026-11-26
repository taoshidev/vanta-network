from enum import Enum


class EntityRole(str, Enum):
    """What an entity hotkey may do. An entity holds one or both roles, each added through entity
    registration for its own fee (ValiConfig.ENTITY_REGISTRATION_FEE or
    ValiConfig.PERMISSIONLESS_ENTITY_MINER_REGISTRATION_FEE)."""
    ENTITY_MINER = "entity_miner"                                 # creates subaccounts under synthetic hotkeys
    PERMISSIONLESS_ENTITY_MINER = "permissionless_entity_miner"   # relays orders signed by traders with their own hotkeys

    @staticmethod
    def is_valid(role: str) -> bool:
        """True if `role` is a valid EntityRole value."""
        if not isinstance(role, str):
            return False
        return role in {r.value for r in EntityRole}
