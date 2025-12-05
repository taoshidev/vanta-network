# developer: trdougherty
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import ValiConfig


class PositionFiltering:
    @staticmethod
    def filter_single_miner(
            positions: list[Position],
            evaluation_time_ms: int,
            lookback_time_ms: int = None
    ) -> list[Position]:
        """
        Restricts to positions which were closed in the prior lookback window
        or positions that are still open and have a return_at_close less than 1
        """
        subset_positions = []
        if not positions:
            return subset_positions

        if lookback_time_ms is None:
            lookback_time_ms = ValiConfig.TARGET_LEDGER_WINDOW_MS

        lookback_threshold_ms = evaluation_time_ms - lookback_time_ms

        for position in positions:
            if position.is_closed_position:
                # Include closed positions within the lookback window
                if position.open_ms >= lookback_threshold_ms:
                    subset_positions.append(position)
            else:
                # Include open positions with return_at_close < 1
                if position.return_at_close < 1:
                    subset_positions.append(position)

        return subset_positions

    @staticmethod
    def filter(
            positions: dict[str, list[Position]],
            evaluation_time_ms: int,
            lookback_time_ms: int = None
    ) -> dict[str, list[Position]]:
        """
        Restricts to positions which were closed in the prior lookback window
        """
        updated_positions = {}

        for miner_hotkey, miner_positions in positions.items():
            updated_positions[miner_hotkey] = PositionFiltering.filter_single_miner(
                miner_positions,
                evaluation_time_ms,
                lookback_time_ms
            )

        return updated_positions

    @staticmethod
    def filter_recent(
            positions: dict[str, list[Position]],
            evaluation_time_ms: int,
            lookback_time_ms: int = None,
            lookback_recent_time_ms: int = None
    ) -> dict[str, list[Position]]:
        """
        Restricts to positions which were closed in the prior lookback window
        """
        updated_positions = {}

        if lookback_time_ms is None:
            lookback_time_ms = ValiConfig.TARGET_LEDGER_WINDOW_MS

        if lookback_recent_time_ms is None:
            lookback_recent_time_ms = ValiConfig.RETURN_SHORT_LOOKBACK_TIME_MS

        lookback_recent_threshold_ms = evaluation_time_ms - lookback_recent_time_ms

        for miner_hotkey, miner_positions in positions.items():
            filtered_miner_positions = PositionFiltering.filter_single_miner(
                miner_positions,
                evaluation_time_ms,
                lookback_time_ms
            )

            recent_filtered_miner_positions = [position for position in filtered_miner_positions if position.close_ms >= lookback_recent_threshold_ms]
            updated_positions[miner_hotkey] = recent_filtered_miner_positions

        return updated_positions

    @staticmethod
    def filter_positions_for_duration(positions: list[Position]):
        """
        Filter out positions that are not at least the minimum position duration.
        """
        filtered_positions = []
        for position in positions:
            if not position.is_closed_position:
                continue

            if position.close_ms - position.open_ms < ValiConfig.MINIMUM_POSITION_DURATION_MS:
                continue

            filtered_positions.append(position)
        return filtered_positions

