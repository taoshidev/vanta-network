import json

from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.price_fetcher.live_price_server import LivePriceFetcherServer

positions_to_snap = [
    {
        # XAUUSD — down validator LONG order + running validator FLAT order
        # LONG entry price differs slightly (4781.7525 vs 4781.7625), so average_entry_price
        # and realized_pnl are recomputed from the down validator's entry.
        "miner_hotkey": "5EPeU7Y8bqokEVf31ZWPZkP3F7Kv1v3ALuhnpp5T5Fvfjp85_96",
        "position_uuid": "ccd7615f-90c0-488c-b93d-d493912b847d",
        "open_ms": 1775748639783,
        "trade_pair": ["XAUUSD", "XAU/USD", 7e-05, 0.1, 4],
        "orders": [
            {
                "order_type": "LONG",
                "leverage": 0.7650804000000001,
                "value": 38254.020000000004,
                "quantity": 0.08,
                "execution_type": "MARKET",
                "limit_price": None,
                "stop_loss": None,
                "take_profit": None,
                "stop_price": None,
                "stop_condition": None,
                "trailing_stop": None,
                "bracket_orders": None,
                "price": 4781.7525000000005,
                "bid": 4781.255,
                "ask": 4782.25,
                "slippage": 0.0001,
                "quote_usd_rate": 1.0,
                "usd_base_rate": 0.0002091283478180855,
                "processed_ms": 1775748639783,
                "order_uuid": "ccd7615f-90c0-488c-b93d-d493912b847d",
                "price_sources": [
                    {
                        "source": "Polygon_ws",
                        "timespan_ms": 0,
                        "open": 4781.7525000000005,
                        "close": 4781.7525000000005,
                        "vwap": 4781.35,
                        "high": 4781.7525000000005,
                        "low": 4781.7525000000005,
                        "start_ms": 1775748639999,
                        "websocket": True,
                        "lag_ms": 216,
                        "bid": 4781.255,
                        "ask": 4782.25
                    },
                    {
                        "source": "Tiingo_ws",
                        "timespan_ms": 0,
                        "open": 4781.08,
                        "close": 4781.08,
                        "vwap": 4781.08,
                        "high": 4781.08,
                        "low": 4781.08,
                        "start_ms": 1775748640000,
                        "websocket": True,
                        "lag_ms": 217,
                        "bid": 4781.08,
                        "ask": 4782.18
                    }
                ],
                "src": 0,
                "margin_loan": 0.0,
                "is_hl_taker": None
            },
            {
                "order_type": "FLAT",
                "leverage": -0.7649728,
                "value": -38254.100000000006,
                "quantity": -0.08,
                "execution_type": "MARKET",
                "limit_price": None,
                "stop_loss": None,
                "take_profit": None,
                "stop_price": None,
                "stop_condition": None,
                "trailing_stop": None,
                "bracket_orders": None,
                "price": 4770.565,
                "bid": 4770.0599999999995,
                "ask": 4771.07,
                "slippage": 0.0001,
                "quote_usd_rate": 1.0,
                "usd_base_rate": 0.0002096187768115517,
                "processed_ms": 1775763684973,
                "order_uuid": "637d4bc7-dddb-4830-be18-e163d0b794df",
                "price_sources": [
                    {
                        "source": "Polygon_ws",
                        "timespan_ms": 0,
                        "open": 4770.565,
                        "close": 4770.565,
                        "vwap": 4770.05,
                        "high": 4770.565,
                        "low": 4770.565,
                        "start_ms": 1775763684999,
                        "websocket": True,
                        "lag_ms": 26,
                        "bid": 4770.0599999999995,
                        "ask": 4771.07
                    },
                    {
                        "source": "Tiingo_ws",
                        "timespan_ms": 0,
                        "open": 4770.36,
                        "close": 4770.36,
                        "vwap": 4770.36,
                        "high": 4770.36,
                        "low": 4770.36,
                        "start_ms": 1775763716000,
                        "websocket": True,
                        "lag_ms": 31027,
                        "bid": 4770.36,
                        "ask": 4771.46
                    }
                ],
                "src": 0,
                "margin_loan": -0.0,
                "is_hl_taker": None
            }
        ],
        # average_entry_price = 4781.7525 * 1.0001 = 4782.230675250001
        # exit_price = 4770.565 * (1 - 0.0001) = 4770.0879435
        # realized_pnl = -1 * (4770.0879435 - 4782.230675250001) * (-0.08 * 100) = -97.14185400000800
        # current_return = 1 + (-97.14185400000800 / 50000) = 0.9980571629199998
        "current_return": 0.9980571629199998,
        "close_ms": 1775763684973,
        "net_leverage": 0.0,
        "net_value": 0.0,
        "net_quantity": 0.0,
        "return_at_close": 0.9980571629199998,
        "average_entry_price": 4782.230675250001,
        "cumulative_entry_value": 38254.020000000004,
        "account_size": 50000.0,
        "realized_pnl": -97.14185400000800,
        "unrealized_pnl": 0.0,
        "position_type": "FLAT",
        "is_closed_position": True,
        "fee_history": [],
        "is_hl": False,
        "last_stock_split_date": None
    },
    {
        # XAGUSD — down validator LONG order + running validator FLAT order
        # LONG entry price is identical between validators (75.354), so computed fields
        # match the running validator exactly.
        "miner_hotkey": "5EPeU7Y8bqokEVf31ZWPZkP3F7Kv1v3ALuhnpp5T5Fvfjp85_96",
        "position_uuid": "539eb696-b49e-4296-a1eb-c7c791b4952a",
        "open_ms": 1775748632183,
        "trade_pair": ["XAGUSD", "XAG/USD", 7e-05, 0.1, 4],
        "orders": [
            {
                "order_type": "LONG",
                "leverage": 0.59617008,
                "value": 29808.504000000004,
                "quantity": 0.0792,
                "execution_type": "MARKET",
                "limit_price": None,
                "stop_loss": None,
                "take_profit": None,
                "stop_price": None,
                "stop_condition": None,
                "trailing_stop": None,
                "bracket_orders": None,
                "price": 75.354,
                "bid": 75.3342,
                "ask": 75.3738,
                "slippage": 0.0001,
                "quote_usd_rate": 1.0,
                "usd_base_rate": 0.013284799532375057,
                "processed_ms": 1775748632183,
                "order_uuid": "539eb696-b49e-4296-a1eb-c7c791b4952a",
                "price_sources": [
                    {
                        "source": "Polygon_ws",
                        "timespan_ms": 0,
                        "open": 75.354,
                        "close": 75.354,
                        "vwap": 75.3342,
                        "high": 75.354,
                        "low": 75.354,
                        "start_ms": 1775748632000,
                        "websocket": True,
                        "lag_ms": 183,
                        "bid": 75.3342,
                        "ask": 75.3738
                    },
                    {
                        "source": "Tiingo_ws",
                        "timespan_ms": 0,
                        "open": 75.279,
                        "close": 75.279,
                        "vwap": 75.279,
                        "high": 75.279,
                        "low": 75.279,
                        "start_ms": 1775748632000,
                        "websocket": True,
                        "lag_ms": 183,
                        "bid": 75.279,
                        "ask": 75.425
                    }
                ],
                "src": 0,
                "margin_loan": 0.0,
                "is_hl_taker": None
            },
            {
                "order_type": "FLAT",
                "leverage": -0.5962096800000001,
                "value": -29840.184000000005,
                "quantity": -0.0792,
                "execution_type": "MARKET",
                "limit_price": None,
                "stop_loss": None,
                "take_profit": None,
                "stop_price": None,
                "stop_condition": None,
                "trailing_stop": None,
                "bracket_orders": None,
                "price": 75.7852,
                "bid": 75.7852,
                "ask": 75.8298,
                "slippage": 0.0001,
                "quote_usd_rate": 1.0,
                "usd_base_rate": 0.013195188506463004,
                "processed_ms": 1775763691624,
                "order_uuid": "5cebe32f-7d39-47a0-b12d-0fbb3022aaf0",
                "price_sources": [
                    {
                        "source": "Polygon_ws",
                        "timespan_ms": 0,
                        "open": 75.7852,
                        "close": 75.7852,
                        "vwap": 75.7852,
                        "high": 75.7852,
                        "low": 75.7852,
                        "start_ms": 1775763691999,
                        "websocket": True,
                        "lag_ms": 375,
                        "bid": 75.7852,
                        "ask": 75.8298
                    },
                    {
                        "source": "Tiingo_ws",
                        "timespan_ms": 0,
                        "open": 75.743,
                        "close": 75.743,
                        "vwap": 75.743,
                        "high": 75.743,
                        "low": 75.743,
                        "start_ms": 1775763717000,
                        "websocket": True,
                        "lag_ms": 25376,
                        "bid": 75.743,
                        "ask": 75.889
                    }
                ],
                "src": 0,
                "margin_loan": -0.0,
                "is_hl_taker": None
            }
        ],
        # average_entry_price = 75.354 * 1.0001 = 75.3615354
        # exit_price = 75.7852 * (1 - 0.0001) = 75.77762148
        # realized_pnl = -1 * (75.77762148 - 75.3615354) * (-0.0792 * 5000) = 164.77008768
        # current_return = 1 + (164.77008768 / 50000) = 1.0032954017536002
        "current_return": 1.0032954017536002,
        "close_ms": 1775763691624,
        "net_leverage": 0.0,
        "net_value": 0.0,
        "net_quantity": 0.0,
        "return_at_close": 1.0032954017536002,
        "average_entry_price": 75.3615354,
        "cumulative_entry_value": 29808.504000000004,
        "account_size": 50000.0,
        "realized_pnl": 164.77008768,
        "unrealized_pnl": 0.0,
        "position_type": "FLAT",
        "is_closed_position": True,
        "fee_history": [],
        "is_hl": False,
        "last_stock_split_date": None
    },
]

if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    lpf = LivePriceFetcherServer(secrets, disable_ws=True)
    for i, position_json in enumerate(positions_to_snap):
        # build the positions as the order edits did not propagate to position-level attributes.
        pos = Position(**position_json)
        pos.rebuild_position_with_updated_orders(lpf)
        positions_to_snap[i] = pos.model_dump()

    for position_json in positions_to_snap:
        pos = Position(**position_json)
        pos.rebuild_position_with_updated_orders(lpf)
        assert pos.is_closed_position
        #print(pos.to_copyable_str())
        str_to_write = json.dumps(pos, cls=CustomEncoder)

        print(pos.model_dump_json(), '\n', str_to_write)
