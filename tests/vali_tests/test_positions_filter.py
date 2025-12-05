# developer: trdougherty

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.vali_dataclasses.position import Position
from vali_objects.position_management.position_utils import PositionFiltering
from vali_objects.vali_config import TradePair


class TestPositionFiltering(TestBase):

    def setUp(self):
        super().setUp()

        # Default constants
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_LOOKBACK_WINDOW = 1000
        self.DEFAULT_LOOKBACK_WINDOW_RECENT = self.DEFAULT_LOOKBACK_WINDOW // 2
        self.DEFAULT_EVALUATION_TIME = 2000
        self.DEFAULT_OPEN_MS = self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW  # 1000
        self.DEFAULT_OLD_CLOSE_MS = self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW - 1  # 999
        self.DEFAULT_NEW_CLOSE_MS = self.DEFAULT_EVALUATION_TIME  # 2000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

        # Helper function to create positions
        def create_position(uuid, open_ms, close_ms=None, return_at_close=1.01, is_closed=True):
            return Position(
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                position_uuid=uuid,
                open_ms=open_ms,
                close_ms=close_ms if close_ms is not None else self.DEFAULT_NEW_CLOSE_MS,
                return_at_close=return_at_close,
                is_closed_position=is_closed,
                trade_pair=self.DEFAULT_TRADE_PAIR,
            )

        # Old positions (open_ms < lookback window)
        self.p1 = create_position("p1", open_ms=0, close_ms=self.DEFAULT_OLD_CLOSE_MS, return_at_close=1.1)
        self.p2 = create_position("p2", open_ms=0, close_ms=self.DEFAULT_OLD_CLOSE_MS, return_at_close=0.9)
        self.p3 = create_position("p3", open_ms=0, close_ms=None, return_at_close=1.1, is_closed=False)
        self.p4 = create_position("p4", open_ms=0, close_ms=None, return_at_close=0.9, is_closed=False)

        # Old positions with close times within the recent lookback window
        self.p5 = create_position("p5", open_ms=0, close_ms=self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW_RECENT, return_at_close=1.1)
        self.p6 = create_position("p6", open_ms=0, close_ms=self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW_RECENT, return_at_close=0.9)

        # New positions (open_ms within the lookback window)
        self.p7 = create_position("p7", open_ms=self.DEFAULT_OPEN_MS, close_ms=self.DEFAULT_NEW_CLOSE_MS, return_at_close=1.1)
        self.p8 = create_position("p8", open_ms=self.DEFAULT_OPEN_MS, close_ms=self.DEFAULT_NEW_CLOSE_MS, return_at_close=0.9)
        self.p9 = create_position("p9", open_ms=self.DEFAULT_OPEN_MS, close_ms=None, return_at_close=1.1, is_closed=False)
        self.p10 = create_position("p10", open_ms=self.DEFAULT_OPEN_MS, close_ms=None, return_at_close=0.9, is_closed=False)

        # New positions with close times within the recent lookback window
        self.p11 = create_position("p11", open_ms=self.DEFAULT_OPEN_MS, close_ms=self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW_RECENT, return_at_close=1.1)
        self.p12 = create_position("p12", open_ms=self.DEFAULT_OPEN_MS, close_ms=self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW_RECENT, return_at_close=0.9)

        # Recent positions (open_ms within the recent lookback window)
        self.p13 = create_position("p13", open_ms=self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW_RECENT, close_ms=self.DEFAULT_NEW_CLOSE_MS, return_at_close=1.1)
        self.p14 = create_position("p14", open_ms=self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW_RECENT, close_ms=self.DEFAULT_NEW_CLOSE_MS, return_at_close=0.9)
        self.p15 = create_position("p15", open_ms=self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW_RECENT, close_ms=None, return_at_close=1.1, is_closed=False)
        self.p16 = create_position("p16", open_ms=self.DEFAULT_EVALUATION_TIME - self.DEFAULT_LOOKBACK_WINDOW_RECENT, close_ms=None, return_at_close=0.9, is_closed=False)

        # Groupings of positions
        self.old_positions = [self.p1, self.p2, self.p3, self.p4]
        self.old_recent_close_positions = [self.p5, self.p6]
        self.new_positions = [self.p7, self.p8, self.p9, self.p10]
        self.new_recent_close_positions = [self.p11, self.p12]
        self.recent_positions = [self.p13, self.p14, self.p15, self.p16]

    def test_old_positions(self):
        positions = self.old_positions + self.old_recent_close_positions

        # Expected to keep: p4 (losing old open)
        keeping_positions = [self.p4]

        # Expected to filter: p1 (winning old close), p2 (losing old close), p3 (winning old open),
        # p5 (winning old, recent close), p6 (losing old, recent close)
        filtering_positions = [self.p1, self.p2, self.p3, self.p5, self.p6]

        filtered_positions = PositionFiltering.filter_single_miner(
            positions,
            evaluation_time_ms=self.DEFAULT_EVALUATION_TIME,
            lookback_time_ms=self.DEFAULT_LOOKBACK_WINDOW,
        )

        for position in keeping_positions:
            self.assertIn(position, filtered_positions)

        for position in filtering_positions:
            self.assertNotIn(position, filtered_positions)

    def test_new_positions(self):
        """New, meaning any normal positions, not necessarily opened recently but within the lookback window"""
        positions = self.new_positions + self.new_recent_close_positions

        # Expected to keep: p7 (winning new close), p8 (losing new close), p10 (losing new open), p11 (winning new, recent close)
        keeping_positions = [self.p7, self.p8, self.p10, self.p11]

        # Expected to filter: p9 (winning new open)
        filtering_positions = [self.p9]

        filtered_positions = PositionFiltering.filter_single_miner(
            positions,
            evaluation_time_ms=self.DEFAULT_EVALUATION_TIME,
            lookback_time_ms=self.DEFAULT_LOOKBACK_WINDOW,
        )

        for position in keeping_positions:
            self.assertIn(position, filtered_positions)

        for position in filtering_positions:
            self.assertNotIn(position, filtered_positions)


    def test_recent_positions(self):
        """Recent positions, which should only keep positions closed within the recent lookback window or open losing"""
        positions = self.new_positions + self.new_recent_close_positions + self.recent_positions

        # Expected to keep: p13 (winning recent close), p14 (losing recent close), p16 (losing recent open)
        keeping_positions = [self.p7, self.p8, self.p10, self.p11, self.p13, self.p14, self.p16]

        # Expected to filter: p15 (winning recent open)
        filtering_positions = [self.p9, self.p15]

        filtered_positions = PositionFiltering.filter_single_miner(
            positions,
            evaluation_time_ms=self.DEFAULT_EVALUATION_TIME,
            lookback_time_ms=self.DEFAULT_LOOKBACK_WINDOW,
        )

        for position in keeping_positions:
            self.assertIn(position, filtered_positions)

        for position in filtering_positions:
            self.assertNotIn(position, filtered_positions)

    def test_filter_miner_positions(self):
        positions = self.old_positions + self.old_recent_close_positions + self.new_positions + self.new_recent_close_positions + self.recent_positions

        # Expected to keep: p4 (losing old open), p7 (winning new close), p8 (losing new close),
        # p10 (losing new open), p11 (winning new, recent close), p12 (losing new, recent close),
        # p13 (winning recent close), p14 (losing recent close), p16 (losing recent open)
        keeping_positions = [self.p4, self.p7, self.p8, self.p10, self.p11, self.p12, self.p13, self.p14, self.p16]

        # Expected to filter: p1 (winning old close), p2 (losing old close), p3 (winning old open),
        # p5 (winning old, recent close), p6 (losing old, recent close), p9 (winning new open),
        # p15 (winning recent open)
        filtering_positions = [self.p1, self.p2, self.p3, self.p5, self.p6, self.p9, self.p15]

        # First, ensure that all the positions we're looking for are found within the lists
        for position in keeping_positions:
            self.assertIn(position, positions)

        for position in filtering_positions:
            self.assertIn(position, positions)

        filtered_positions = PositionFiltering.filter_single_miner(
            positions,
            evaluation_time_ms=self.DEFAULT_EVALUATION_TIME,
            lookback_time_ms=self.DEFAULT_LOOKBACK_WINDOW,
        )

        for position in keeping_positions:
            self.assertIn(position, filtered_positions)

        for position in filtering_positions:
            self.assertNotIn(position, filtered_positions)

    def test_open_position_with_loss(self):
        """Test that open positions with return_at_close < 1 are included in the results."""
        # Create an open position with a return_at_close less than 1
        open_losing_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="open_losing_position",
            open_ms=self.DEFAULT_OPEN_MS,
            close_ms=None,  # Position is still open
            return_at_close=0.8,  # Losing position
            is_closed_position=False,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )

        positions = [open_losing_position]

        # Run the filtering
        filtered_positions = PositionFiltering.filter_single_miner(
            positions,
            evaluation_time_ms=self.DEFAULT_EVALUATION_TIME,
            lookback_time_ms=self.DEFAULT_LOOKBACK_WINDOW,
        )

        # Check that the open losing position is included
        self.assertIn(open_losing_position, filtered_positions)
