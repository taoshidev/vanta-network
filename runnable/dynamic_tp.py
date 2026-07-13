#!/usr/bin/env python3
"""
Print all hotkeys that have positions with dynamic (HL) trade pairs and their position UUIDs.

Usage:
    python runnable/print_dynamic_trade_pair_positions.py
"""

from vali_objects.position_management.position_manager import PositionManager
from vali_objects.vali_config import DynamicTradePair


def main():
    position_manager = PositionManager(load_from_disk=True)

    found = False
    for hotkey, positions_dict in position_manager.hotkey_to_positions.items():
        for uuid, position in positions_dict.items():
            if isinstance(position.trade_pair, DynamicTradePair):
                print(f"hotkey={hotkey}  uuid={uuid}  trade_pair={position.trade_pair.trade_pair_id}")
                found = True

    if not found:
        print("No positions with dynamic trade pairs found.")


if __name__ == "__main__":
    main()
