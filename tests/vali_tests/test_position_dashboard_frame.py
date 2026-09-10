"""
The v2 dashboard frame for a position (`Position.to_dashboard`).

WHY THIS EXISTS. The frame carried `nl` (net_leverage) but never `net_value`,
and the two are NOT interchangeable downstream: `net_leverage` is
`net_value / self.account_size` where `account_size` is THIS POSITION's
snapshot (see `Position.update_position_state`), not the subaccount's nominal
account size. A dashboard client that receives only `nl` and multiplies by the
nominal size therefore gets a figure wrong by the ratio between the two —
constant per account, and measured at up to ~16% (worst case $30,385 on a
$221K book) across live mainnet subaccounts on 2026-09-01.

That matters beyond display: `get_max_order_size()` compares the per-pair cap
against `abs(position.net_value)`, so a client sizing from `nl` cannot
reproduce the very cap its orders are clamped by.

Plain `unittest.TestCase` on purpose — this is pure serialization and needs no
orchestrator or server fixtures.
"""
import unittest

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.position import Position


def _position(**overrides) -> Position:
    position = Position(
        miner_hotkey="test_miner",
        position_uuid="test_position",
        open_ms=1_700_000_000_000,
        trade_pair=TradePair.BTCUSD,
        account_size=100_000,
        position_type=OrderType.LONG,
    )
    for key, value in overrides.items():
        setattr(position, key, value)
    return position


class TestPositionDashboardFrame(unittest.TestCase):
    def test_emits_net_value_alongside_net_leverage(self):
        position = _position(
            net_leverage=0.25,
            net_quantity=2.0,
            net_value=123456.78,
        )

        frame = position.to_dashboard(0, filled_orders={}, unfilled_orders={})

        self.assertIn("nv", frame)
        self.assertEqual(frame["nv"], 123456.78)
        # Both facts must travel together — shipping `nl` alone is the defect.
        self.assertEqual(frame["nl"], 0.25)

    def test_net_value_is_not_derivable_from_net_leverage_and_nominal_size(self):
        """
        The regression this field exists to prevent.

        Here the position's own account_size snapshot ($115,473) differs from
        the subaccount's nominal $100,000 — an account up ~15%. Reconstructing
        the value as `nominal_account_size * nl` understates it by that same
        ratio, which is exactly the error observed in production.
        """
        net_value = 86_604.75
        position_account_size = 115_473.0
        position = _position(
            account_size=position_account_size,
            net_quantity=1.0,
            net_value=net_value,
            net_leverage=net_value / position_account_size,
        )

        frame = position.to_dashboard(0, filled_orders={}, unfilled_orders={})

        reconstructed = 100_000 * frame["nl"]
        self.assertNotAlmostEqual(reconstructed, net_value, delta=1.0)
        # The emitted value is exact regardless of the snapshot difference.
        self.assertEqual(frame["nv"], net_value)

    def test_omits_net_value_when_zero(self):
        """
        Same truthiness gate as `nl`: a flat/closed position carries
        net_value 0.0 and must not grow every frame with a dead key.
        """
        position = _position(net_leverage=0.0, net_quantity=0.0, net_value=0.0)

        frame = position.to_dashboard(0, filled_orders={}, unfilled_orders={})

        self.assertNotIn("nv", frame)
        self.assertNotIn("nl", frame)

    def test_short_position_keeps_the_sign(self):
        """
        `net_value` is signed for a SHORT. Consumers compare `abs(net_value)`
        (as `get_max_order_size` does); the sign is preserved rather than
        normalised so direction stays recoverable from the frame alone.
        """
        position = _position(
            position_type=OrderType.SHORT,
            net_quantity=-2.0,
            net_value=-50_000.0,
            net_leverage=-0.5,
        )

        frame = position.to_dashboard(0, filled_orders={}, unfilled_orders={})

        self.assertEqual(frame["nv"], -50_000.0)
        self.assertEqual(abs(frame["nv"]), 50_000.0)


if __name__ == "__main__":
    unittest.main()
