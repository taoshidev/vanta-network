import hashlib
import pickle
import time
from typing import Union

import numpy as np

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    TP_ID_PORTFOLIO,
    PerfCheckpoint,
    PerfLedger,
)


def get_time_in_range(percent, start, end):
    return int(start + ((end - start) * percent))


def hash_object(obj):
    serialized_obj = pickle.dumps(obj)
    hash_obj = hashlib.sha256()
    hash_obj.update(serialized_obj)
    hashed_str = hash_obj.hexdigest()
    return hashed_str[:10]


def order_generator(
        order_type=OrderType.LONG,
        leverage=1.0,
        n_orders: int = 10,
        processed_ms: int = 1710521764446,
) -> list[Order]:
    order_list = []
    for _ in range(n_orders):
        sample_order = Order(
            order_type=order_type,
            leverage=leverage,
            price=3000,
            trade_pair=TradePair.BTCUSD,
            processed_ms=processed_ms,
            order_uuid="1000",
        )

        order_list.append(sample_order)

    return order_list


def position_generator(
        open_time_ms,
        trade_pair,
        close_time_ms: Union[None, int] = None,
        return_at_close=1.0,
        orders: list[Order] = [],
        miner_hotkey: str = 'miner0',
):
    generated_position = Position(
        miner_hotkey=miner_hotkey,
        position_uuid=hash_object((
            open_time_ms,
            close_time_ms,
            return_at_close,
            trade_pair,
        )),
        orders=orders,
        net_leverage=0.0,
        open_ms=open_time_ms,
        trade_pair=trade_pair,
    )

    if close_time_ms is not None:
        generated_position.close_out_position(
            close_ms=close_time_ms,
        )
        generated_position.return_at_close = return_at_close

    return generated_position


def generate_ledger(
        value=None,
        start_time=0,
        nterms=None,
        end_time=ValiConfig.TARGET_LEDGER_WINDOW_MS,
        gain=None,
        loss=None,
        open_ms=None,
        mdd=1.0,
):
    # Check for invalid input combinations
    if value is None and (gain is None or loss is None):
        raise ValueError("Either 'value' or both 'gain' and 'loss' must be provided")

    # If value is provided and gain/loss are not, set gain/loss based on value
    if value is not None and gain is None and loss is None:
        gain = value
        loss = -value

    # If gain or loss is provided without value, just use the provided gain/loss
    if gain is not None and loss is not None and value is None:
        pass  # gain and loss are already set correctly

    # Determine the number of terms if not provided
    if nterms is None:
        nterms = (end_time - start_time) // ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    # Generate checkpoint times
    checkpoint_times = [start_time + i * ValiConfig.TARGET_CHECKPOINT_DURATION_MS for i in range(nterms)]
    checkpoint_time_accumulation = np.diff(checkpoint_times, prepend=start_time)

    checkpoint_list = []
    for i in range(len(checkpoint_times)):
        if open_ms is None:
            checkpoint_open_ms = checkpoint_time_accumulation[i]
        else:
            checkpoint_open_ms = open_ms

        checkpoint_list.append(
            checkpoint_generator(
                last_update_ms=checkpoint_times[i],
                gain=gain,
                loss=loss,
                prev_portfolio_ret=1.0,
                open_ms=checkpoint_open_ms,
                accum_ms=checkpoint_open_ms,
                mdd=mdd,
            ),
        )

    base_ledger = ledger_generator(checkpoints=checkpoint_list)
    return {TP_ID_PORTFOLIO: base_ledger, TradePair.BTCUSD.trade_pair_id: base_ledger}


def ledger_generator(
        target_cp_duration: int = ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
        target_ledger_window_ms: float = ValiConfig.TARGET_LEDGER_WINDOW_MS,
        checkpoints=None,
):
    if checkpoints is None:
        checkpoints = []
    return PerfLedger(
        target_cp_duration_ms=target_cp_duration,
        target_ledger_window_ms=target_ledger_window_ms,
        cps=checkpoints,
    )


def checkpoint_generator(
        last_update_ms: int = 0,
        prev_portfolio_ret: float = 1.0,
        accum_ms: int = 0,
        open_ms: int = 0,
        n_updates: int = 0,
        gain: float = 0.0,
        loss: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        pnl_gain: float = 0.0,
        pnl_loss: float = 0.0,
        mdd: float = 1.0,
):
    if pnl_gain != 0.0 or pnl_loss != 0.0:
        realized_pnl = pnl_gain + pnl_loss

    return PerfCheckpoint(
        last_update_ms=last_update_ms,
        prev_portfolio_ret=prev_portfolio_ret,
        accum_ms=accum_ms,
        open_ms=open_ms,
        n_updates=n_updates,
        gain=gain,
        loss=loss,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        mdd=mdd,
    )

def generate_winning_ledger(start, end):
    # Designed with challenge period in mind
    btc_ledger = generate_ledger(start_time=start, end_time=end, gain=0.1, loss=-0.08, mdd=0.99)
    portfolio_ledger = generate_ledger(start_time=start, end_time=end, gain=0.1, loss=-0.08, mdd=0.99)

    return {
            TP_ID_PORTFOLIO: portfolio_ledger[TP_ID_PORTFOLIO],
            "BTCUSD": btc_ledger[TP_ID_PORTFOLIO]
            }

def generate_losing_ledger(start, end):
    # Designed with challenge period in mind
    btc_ledger = generate_ledger(start_time=start, end_time=end, gain=0.1, loss=-0.2, mdd=(1 - (ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE / 100)) - 0.01)
    portfolio_ledger = generate_ledger(start_time=start, end_time=end, gain=0.1, loss=-0.2, mdd=(1 - (ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE / 100)) - 0.01)

    return {
        TP_ID_PORTFOLIO: portfolio_ledger[TP_ID_PORTFOLIO],
        "BTCUSD": btc_ledger[TP_ID_PORTFOLIO]
    }

def create_daily_checkpoints_with_pnl(pnl_values: list[float]) -> PerfLedger:
        """Helper method to create checkpoints for complete days with specific PnL values"""
        checkpoints = []
        # Use fixed timestamp for deterministic tests (2024-01-01 00:00:00 UTC)
        # This ensures tests are reproducible regardless of when they run
        current_time_ms = 1735689600000  # 2024-01-01 00:00:00 UTC (midnight)
        checkpoint_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        checkpoints_per_day = int(ValiConfig.DAILY_CHECKPOINTS)

        for day_idx, daily_pnl in enumerate(pnl_values):
            # Split daily PnL across checkpoints for the day
            pnl_per_checkpoint = daily_pnl / checkpoints_per_day
            pnl_gain = pnl_per_checkpoint if pnl_per_checkpoint > 0 else 0
            pnl_loss = pnl_per_checkpoint if pnl_per_checkpoint < 0 else 0

            # Calculate the start of this day (since current_time_ms is already midnight UTC,
            # this gives us midnight of each subsequent day)
            day_start_ms = current_time_ms + (day_idx * 24 * 60 * 60 * 1000)
            
            for cp_idx in range(checkpoints_per_day):
                # Position checkpoints within the same day
                # Start each checkpoint at the beginning of its interval within the day
                checkpoint_start_ms = day_start_ms + (cp_idx * checkpoint_duration_ms)
                checkpoint_end_ms = checkpoint_start_ms + checkpoint_duration_ms

                cp = PerfCheckpoint(
                    last_update_ms=checkpoint_end_ms,
                    prev_portfolio_ret=1.0,
                    accum_ms=checkpoint_duration_ms,  # Complete checkpoint
                    pnl_gain=pnl_gain,
                    pnl_loss=pnl_loss,
                    gain=0.01,  # Small positive gain for valid checkpoint
                    loss=0.0,
                    mdd=0.95  # No significant drawdown
                )
                checkpoints.append(cp)

        return PerfLedger(cps=checkpoints)
