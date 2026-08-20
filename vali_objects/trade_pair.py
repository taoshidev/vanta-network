# developer: Taoshi
"""TradePair enum and supporting types/constants.

Split out of vali_config.py so the trade-pair domain has its own module.
vali_config.py re-exports the public classes from here for backwards compatibility.
"""
from collections import defaultdict
from enum import Enum
from typing import NamedTuple

class TradePairCategory(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    INDICES = "indices"
    EQUITIES = "equities"
    COMMODITIES = "commodities"

class TradePairSource(str, Enum):
    VANTA = "vanta"
    HYPERLIQUID = "hyperliquid"


class InstrumentType(str, Enum):
    SPOT = "spot"
    PERP = "perp"

class SubaccountTierBaseLeverage(NamedTuple):
    """Tagged wrapper for the per-pair Tier-1 base used by subaccount tier dispatch.

    The dedicated type lets the TradePair.subaccount_tier_base_leverage property locate
    it via isinstance scan over the value list, independent of position. NamedTuple keeps
    the wrapper distinct from the raw fees/min_leverage/max_leverage floats —
    `isinstance(_, float)` returns False and hash/equality don't collide with regular
    floats. Unwrap via `.value`.
    """
    value: float


class TradePairSubcategory(str, Enum):
    """
    All concrete sub‑category enums must set `ASSET_CLASS`
    to one of the TradePairCategory members.
    """
    @property
    def asset_class(self) -> TradePairCategory:
        raise NotImplementedError("Subclasses must implement the asset_class property.")

class ForexSubcategory(TradePairSubcategory):
    G1 = "forex_group1"
    G2 = "forex_group2"
    G3 = "forex_group3"
    G4 = "forex_group4"
    G5 = "forex_group5"

    @property
    def asset_class(self) -> TradePairCategory:
        return TradePairCategory.FOREX

class CryptoSubcategory(TradePairSubcategory):
    MAJORS = "crypto_majors"
    ALTS = "crypto_alts"

    @property
    def asset_class(self) -> TradePairCategory:
        return TradePairCategory.CRYPTO


class EquitiesSubcategory(TradePairSubcategory):
    LARGE_CAP = "equities_large_cap"
    MID_CAP = "equities_mid_cap"
    SMALL_CAP = "equities_small_cap"

    @property
    def asset_class(self) -> TradePairCategory:
        return TradePairCategory.EQUITIES


class IndicesSubcategory(TradePairSubcategory):
    GLOBAL = "indices_global"
    REGIONAL = "indices_regional"
    SECTOR = "indices_sector"

    @property
    def asset_class(self) -> TradePairCategory:
        return TradePairCategory.INDICES

# Positional leverage limits used in TradePair definitions below.
CRYPTO_MIN_LEVERAGE = 0.01
CRYPTO_MAX_LEVERAGE = 2.5
FOREX_MIN_LEVERAGE = 0.1
FOREX_MAX_LEVERAGE = 10
INDICES_MIN_LEVERAGE = 0.1
INDICES_MAX_LEVERAGE = 5
EQUITIES_MIN_LEVERAGE = 0.01
EQUITIES_MAX_LEVERAGE = 2
COMMODITIES_MIN_LEVERAGE = 0.05
COMMODITIES_MAX_LEVERAGE = 2

# HL/HS leverage caps used in TradePair definitions below.
HS_MIN_LEVERAGE = 0.01
HS_MAX_LEVERAGE = 1.0

# (taker, maker) for HL-sourced pairs. Priced below HL's own schedule
HL_FEE_BY_CATEGORY = {
    TradePairCategory.CRYPTO:      (0.0003,  0.0003),   # 3 bps
    TradePairCategory.EQUITIES:    (0.0001,  0.0001),   # 1 bp
    TradePairCategory.COMMODITIES: (0.00005, 0.00005),  # 0.5 bps
    TradePairCategory.INDICES:     (0, 0),
    TradePairCategory.FOREX:       (0, 0),
}

# Vanta fee constants
TRANSACTION_FEE_RATE = {
    TradePairCategory.CRYPTO:      0.0003,   # 3 bps
    TradePairCategory.EQUITIES:    0.0001,   # 1 bp
    TradePairCategory.COMMODITIES: 0.00005,  # 0.5 bps
    TradePairCategory.FOREX:       0,
    TradePairCategory.INDICES:     0,
}

CARRY_FEE_RATE_PER_INTERVAL = {
    TradePairCategory.CRYPTO:      0.0001,        # 10.95% annual / (365 * 3)
    TradePairCategory.FOREX:       0.0000821918,   # 3% annual / 365
    TradePairCategory.INDICES:     0.0001438356,   # 5.25% annual / 365
    TradePairCategory.COMMODITIES: 0,              # HL funding used instead
    TradePairCategory.EQUITIES:    0,              # equity-specific rates below
}

ANNUAL_STOCK_BORROW_RATE    = 0.03   # 3% — short equity stock-borrow fee
DAILY_STOCK_BORROW_RATE     = ANNUAL_STOCK_BORROW_RATE / 365

ANNUAL_MARGIN_INTEREST_RATE = 0.066  # 6.6% — long equity margin interest
DAILY_MARGIN_INTEREST_RATE  = ANNUAL_MARGIN_INTEREST_RATE / 365

# Trade-pair id sets used by TradePair.is_blocked / is_flat_only.
FLAT_ONLY_TRADE_PAIR_IDS = {}
BLOCKED_TRADE_PAIR_IDS = {
    'SPX', 'DJI', 'NDX', 'VIX', 'FTSE', 'GDAXI',  # Indices
    'USDMXN',
    'PAXGUSDC',      # Gold; kept GOLDUSDC
    'BRENTOILUSDC',  # Oil; kept WTIOILUSDC
    'XAGUSD', 'XAUUSD',  # replaced with GOLDUSDC, SILVERUSDC
    'TONUSDC',  # Delisted from Hyperliquid

    # All vanta native crypto pairs deprecated for corresponding USDC pairs
    'BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD',
    'DOGEUSD', 'ADAUSD', 'TAOUSD', 'HYPEUSD',
    'ZECUSD', 'BCHUSD', 'LINKUSD', 'XMRUSD',
    'LTCUSD',
    
    'NSA'  # de-listed on 2026-07-22 NOTE could potentially delete trade pair
}


class TradePair(Enum):
    # Vanta Native Trade Pairs
    # crypto
    BTCUSD  = ["BTCUSD",  "BTC/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.MAJORS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ETHUSD  = ["ETHUSD",  "ETH/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.MAJORS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SOLUSD  = ["SOLUSD",  "SOL/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XRPUSD  = ["XRPUSD",  "XRP/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DOGEUSD = ["DOGEUSD", "DOGE/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADAUSD  = ["ADAUSD",  "ADA/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TAOUSD  = ["TAOUSD",  "TAO/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HYPEUSD = ["HYPEUSD", "HYPE/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZECUSD  = ["ZECUSD",  "ZEC/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BCHUSD  = ["BCHUSD",  "BCH/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LINKUSD = ["LINKUSD", "LINK/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XMRUSD  = ["XMRUSD",  "XMR/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LTCUSD  = ["LTCUSD",  "LTC/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]

    # forex
    AUDCAD = ["AUDCAD", "AUD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    AUDCHF = ["AUDCHF", "AUD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    AUDUSD = ["AUDUSD", "AUD/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(20.0)]
    AUDJPY = ["AUDJPY", "AUD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    AUDNZD = ["AUDNZD", "AUD/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    CADCHF = ["CADCHF", "CAD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    CADJPY = ["CADJPY", "CAD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    CHFJPY = ["CHFJPY", "CHF/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    EURAUD = ["EURAUD", "EUR/AUD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    EURCAD = ["EURCAD", "EUR/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    EURUSD = ["EURUSD", "EUR/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(20.0)]
    EURCHF = ["EURCHF", "EUR/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    EURGBP = ["EURGBP", "EUR/GBP", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    EURJPY = ["EURJPY", "EUR/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    EURNZD = ["EURNZD", "EUR/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    NZDCAD = ["NZDCAD", "NZD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    NZDCHF = ["NZDCHF", "NZD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    NZDJPY = ["NZDJPY", "NZD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    NZDUSD = ["NZDUSD", "NZD/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(20.0)]
    GBPAUD = ["GBPAUD", "GBP/AUD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    GBPCAD = ["GBPCAD", "GBP/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    GBPCHF = ["GBPCHF", "GBP/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    GBPJPY = ["GBPJPY", "GBP/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    GBPNZD = ["GBPNZD", "GBP/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(10.0)]
    GBPUSD = ["GBPUSD", "GBP/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(20.0)]
    USDCAD = ["USDCAD", "USD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(20.0)]
    USDCHF = ["USDCHF", "USD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(20.0)]
    USDJPY = ["USDJPY", "USD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(20.0)]
    USDMXN = ["USDMXN", "USD/MXN", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]


    # "Commodities" (Bundle with Forex for now)
    XAUUSD = ["XAUUSD", "XAU/USD", 0.00007, COMMODITIES_MIN_LEVERAGE, COMMODITIES_MAX_LEVERAGE, TradePairCategory.FOREX, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    XAGUSD = ["XAGUSD", "XAG/USD", 0.00007, COMMODITIES_MIN_LEVERAGE, COMMODITIES_MAX_LEVERAGE, TradePairCategory.FOREX, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]

    # Equities - Stocks
    # Technology (10)
    NVDA = ["NVDA", "NVDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MSFT = ["MSFT", "MSFT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AAPL = ["AAPL", "AAPL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AVGO = ["AVGO", "AVGO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TSM  = ["TSM",  "TSM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ORCL = ["ORCL", "ORCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMD  = ["AMD",  "AMD",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MU   = ["MU",   "MU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CRM  = ["CRM",  "CRM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UBER = ["UBER", "UBER", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    # Financial Services (5)
    BRK_B = ["BRK_B", "BRK.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JPM   = ["JPM",   "JPM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    V     = ["V",     "V",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MA    = ["MA",    "MA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BAC   = ["BAC",   "BAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    # Consumer Discretionary (5)
    AMZN = ["AMZN", "AMZN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TSLA = ["TSLA", "TSLA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HD   = ["HD",   "HD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BABA = ["BABA", "BABA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SBUX = ["SBUX", "SBUX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    # Communication Services (5)
    GOOGL = ["GOOGL", "GOOGL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    META  = ["META",  "META",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NFLX  = ["NFLX",  "NFLX",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    APP   = ["APP",   "APP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    T     = ["T",     "T",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    # Spot single stocks matching Hyperliquid equity perps (8)
    COIN = ["COIN", "COIN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CRCL = ["CRCL", "CRCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MSTR = ["MSTR", "MSTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PLTR = ["PLTR", "PLTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SNDK = ["SNDK", "SNDK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INTC = ["INTC", "INTC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HOOD = ["HOOD", "HOOD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SPCX = ["SPCX", "SPCX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]

    # Russell 1000 stocks bulk-added by runnable/generate_equity_universe.py (additive: appends new
    # tickers, never touches existing). Per-pair fees/base literals here are hand-editable.
    A      = ["A",      "A",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AA     = ["AA",     "AA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AAL    = ["AAL",    "AAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AAON   = ["AAON",   "AAON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ABBV   = ["ABBV",   "ABBV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ABNB   = ["ABNB",   "ABNB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ABT    = ["ABT",    "ABT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ACGL   = ["ACGL",   "ACGL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ACHC   = ["ACHC",   "ACHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ACI    = ["ACI",    "ACI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ACM    = ["ACM",    "ACM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ACN    = ["ACN",    "ACN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ADBE   = ["ADBE",   "ADBE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ADC    = ["ADC",    "ADC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ADI    = ["ADI",    "ADI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ADM    = ["ADM",    "ADM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ADP    = ["ADP",    "ADP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ADSK   = ["ADSK",   "ADSK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ADT    = ["ADT",    "ADT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AEE    = ["AEE",    "AEE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AEP    = ["AEP",    "AEP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AES    = ["AES",    "AES",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AFG    = ["AFG",    "AFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AFL    = ["AFL",    "AFL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AFRM   = ["AFRM",   "AFRM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AGCO   = ["AGCO",   "AGCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AGNC   = ["AGNC",   "AGNC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AGO    = ["AGO",    "AGO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AIG    = ["AIG",    "AIG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AIT    = ["AIT",    "AIT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AIZ    = ["AIZ",    "AIZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AJG    = ["AJG",    "AJG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AKAM   = ["AKAM",   "AKAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALAB   = ["ALAB",   "ALAB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALB    = ["ALB",    "ALB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALGM   = ["ALGM",   "ALGM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALGN   = ["ALGN",   "ALGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALK    = ["ALK",    "ALK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALL    = ["ALL",    "ALL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALLE   = ["ALLE",   "ALLE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALLY   = ["ALLY",   "ALLY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALNY   = ["ALNY",   "ALNY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ALSN   = ["ALSN",   "ALSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AM     = ["AM",     "AM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMAT   = ["AMAT",   "AMAT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMCR   = ["AMCR",   "AMCR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AME    = ["AME",    "AME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMG    = ["AMG",    "AMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMGN   = ["AMGN",   "AMGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMH    = ["AMH",    "AMH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMKR   = ["AMKR",   "AMKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMP    = ["AMP",    "AMP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMT    = ["AMT",    "AMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AMTM   = ["AMTM",   "AMTM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AN     = ["AN",     "AN",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ANET   = ["ANET",   "ANET",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AON    = ["AON",    "AON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AOS    = ["AOS",    "AOS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    APA    = ["APA",    "APA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    APD    = ["APD",    "APD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    APG    = ["APG",    "APG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    APH    = ["APH",    "APH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    APO    = ["APO",    "APO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    APPF   = ["APPF",   "APPF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    APTV   = ["APTV",   "APTV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AR     = ["AR",     "AR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ARE    = ["ARE",    "ARE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ARES   = ["ARES",   "ARES",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ARMK   = ["ARMK",   "ARMK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ARW    = ["ARW",    "ARW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AS     = ["AS",     "AS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ASH    = ["ASH",    "ASH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ASTS   = ["ASTS",   "ASTS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ATI    = ["ATI",    "ATI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ATO    = ["ATO",    "ATO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ATR    = ["ATR",    "ATR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AU     = ["AU",     "AU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AUR    = ["AUR",    "AUR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AVB    = ["AVB",    "AVB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AVT    = ["AVT",    "AVT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AVTR   = ["AVTR",   "AVTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AVY    = ["AVY",    "AVY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AWI    = ["AWI",    "AWI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AWK    = ["AWK",    "AWK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AXON   = ["AXON",   "AXON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AXP    = ["AXP",    "AXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AXS    = ["AXS",    "AXS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AXTA   = ["AXTA",   "AXTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AYI    = ["AYI",    "AYI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    AZO    = ["AZO",    "AZO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BA     = ["BA",     "BA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BAH    = ["BAH",    "BAH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BALL   = ["BALL",   "BALL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BAM    = ["BAM",    "BAM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BAX    = ["BAX",    "BAX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BBWI   = ["BBWI",   "BBWI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BBY    = ["BBY",    "BBY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BC     = ["BC",     "BC",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BDX    = ["BDX",    "BDX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BEN    = ["BEN",    "BEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BEPC   = ["BEPC",   "BEPC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BFAM   = ["BFAM",   "BFAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BF_A   = ["BF_A",   "BF.A",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BF_B   = ["BF_B",   "BF.B",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BG     = ["BG",     "BG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BHF    = ["BHF",    "BHF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BIIB   = ["BIIB",   "BIIB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BILL   = ["BILL",   "BILL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BIO    = ["BIO",    "BIO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BIRK   = ["BIRK",   "BIRK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BJ     = ["BJ",     "BJ",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BKNG   = ["BKNG",   "BKNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BKR    = ["BKR",    "BKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BLDR   = ["BLDR",   "BLDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BLK    = ["BLK",    "BLK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BLSH   = ["BLSH",   "BLSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BMRN   = ["BMRN",   "BMRN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BMY    = ["BMY",    "BMY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BNY    = ["BNY",    "BNY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BOKF   = ["BOKF",   "BOKF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BPOP   = ["BPOP",   "BPOP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BR     = ["BR",     "BR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BRBR   = ["BRBR",   "BRBR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BRKR   = ["BRKR",   "BRKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BRO    = ["BRO",    "BRO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BROS   = ["BROS",   "BROS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BRX    = ["BRX",    "BRX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BSX    = ["BSX",    "BSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BSY    = ["BSY",    "BSY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BURL   = ["BURL",   "BURL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BWA    = ["BWA",    "BWA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BWXT   = ["BWXT",   "BWXT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BX     = ["BX",     "BX",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BXP    = ["BXP",    "BXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    BYD    = ["BYD",    "BYD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    C      = ["C",      "C",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CACC   = ["CACC",   "CACC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CACI   = ["CACI",   "CACI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CAG    = ["CAG",    "CAG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CAH    = ["CAH",    "CAH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CAI    = ["CAI",    "CAI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CAR    = ["CAR",    "CAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CARR   = ["CARR",   "CARR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CART   = ["CART",   "CART",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CASY   = ["CASY",   "CASY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CAT    = ["CAT",    "CAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CAVA   = ["CAVA",   "CAVA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CB     = ["CB",     "CB",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CBC    = ["CBC",    "CBC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CBOE   = ["CBOE",   "CBOE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CBRE   = ["CBRE",   "CBRE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CBSH   = ["CBSH",   "CBSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CCC    = ["CCC",    "CCC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CCI    = ["CCI",    "CCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CCK    = ["CCK",    "CCK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CCL    = ["CCL",    "CCL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CDNS   = ["CDNS",   "CDNS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CDW    = ["CDW",    "CDW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CE     = ["CE",     "CE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CEG    = ["CEG",    "CEG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CELH   = ["CELH",   "CELH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CERT   = ["CERT",   "CERT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CF     = ["CF",     "CF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CFG    = ["CFG",    "CFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CFR    = ["CFR",    "CFR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CG     = ["CG",     "CG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CGNX   = ["CGNX",   "CGNX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CHD    = ["CHD",    "CHD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CHDN   = ["CHDN",   "CHDN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CHE    = ["CHE",    "CHE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CHH    = ["CHH",    "CHH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CHRD   = ["CHRD",   "CHRD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CHRW   = ["CHRW",   "CHRW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CHTR   = ["CHTR",   "CHTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CHWY   = ["CHWY",   "CHWY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CI     = ["CI",     "CI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CIEN   = ["CIEN",   "CIEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CINF   = ["CINF",   "CINF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CL     = ["CL",     "CL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CLF    = ["CLF",    "CLF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CLH    = ["CLH",    "CLH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CLVT   = ["CLVT",   "CLVT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CLX    = ["CLX",    "CLX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CMCSA  = ["CMCSA",  "CMCSA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CME    = ["CME",    "CME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CMG    = ["CMG",    "CMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CMI    = ["CMI",    "CMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CMS    = ["CMS",    "CMS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CNA    = ["CNA",    "CNA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CNC    = ["CNC",    "CNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CNH    = ["CNH",    "CNH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CNM    = ["CNM",    "CNM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CNP    = ["CNP",    "CNP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CNXC   = ["CNXC",   "CNXC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COF    = ["COF",    "COF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COHR   = ["COHR",   "COHR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COKE   = ["COKE",   "COKE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COLB   = ["COLB",   "COLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COLD   = ["COLD",   "COLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COLM   = ["COLM",   "COLM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COO    = ["COO",    "COO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COP    = ["COP",    "COP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COR    = ["COR",    "COR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CORT   = ["CORT",   "CORT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COST   = ["COST",   "COST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    COTY   = ["COTY",   "COTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CPAY   = ["CPAY",   "CPAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CPB    = ["CPB",    "CPB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CPNG   = ["CPNG",   "CPNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CPRT   = ["CPRT",   "CPRT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CPT    = ["CPT",    "CPT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CR     = ["CR",     "CR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CRBG   = ["CRBG",   "CRBG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CRH    = ["CRH",    "CRH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CRL    = ["CRL",    "CRL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CROX   = ["CROX",   "CROX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CRS    = ["CRS",    "CRS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CRUS   = ["CRUS",   "CRUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CRWD   = ["CRWD",   "CRWD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CSCO   = ["CSCO",   "CSCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CSGP   = ["CSGP",   "CSGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CSL    = ["CSL",    "CSL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CSX    = ["CSX",    "CSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CTAS   = ["CTAS",   "CTAS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CTSH   = ["CTSH",   "CTSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CTVA   = ["CTVA",   "CTVA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CUBE   = ["CUBE",   "CUBE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CUZ    = ["CUZ",    "CUZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CVNA   = ["CVNA",   "CVNA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CVS    = ["CVS",    "CVS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CVX    = ["CVX",    "CVX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CW     = ["CW",     "CW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CWEN   = ["CWEN",   "CWEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CXT    = ["CXT",    "CXT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    CZR    = ["CZR",    "CZR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    D      = ["D",      "D",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DAL    = ["DAL",    "DAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DAR    = ["DAR",    "DAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DASH   = ["DASH",   "DASH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DBX    = ["DBX",    "DBX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DCI    = ["DCI",    "DCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DD     = ["DD",     "DD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DDOG   = ["DDOG",   "DDOG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DDS    = ["DDS",    "DDS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DE     = ["DE",     "DE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DECK   = ["DECK",   "DECK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DELL   = ["DELL",   "DELL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DG     = ["DG",     "DG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DGX    = ["DGX",    "DGX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DHI    = ["DHI",    "DHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DHR    = ["DHR",    "DHR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DINO   = ["DINO",   "DINO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DIS    = ["DIS",    "DIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DJT    = ["DJT",    "DJT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DKNG   = ["DKNG",   "DKNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DKS    = ["DKS",    "DKS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DLB    = ["DLB",    "DLB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DLR    = ["DLR",    "DLR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DLTR   = ["DLTR",   "DLTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DOC    = ["DOC",    "DOC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DOCS   = ["DOCS",   "DOCS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DOCU   = ["DOCU",   "DOCU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DOV    = ["DOV",    "DOV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DOW    = ["DOW",    "DOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DOX    = ["DOX",    "DOX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DPZ    = ["DPZ",    "DPZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DRI    = ["DRI",    "DRI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DRS    = ["DRS",    "DRS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DT     = ["DT",     "DT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DTE    = ["DTE",    "DTE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DTM    = ["DTM",    "DTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DUK    = ["DUK",    "DUK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DUOL   = ["DUOL",   "DUOL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DV     = ["DV",     "DV",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DVA    = ["DVA",    "DVA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DVN    = ["DVN",    "DVN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DXC    = ["DXC",    "DXC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DXCM   = ["DXCM",   "DXCM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EA     = ["EA",     "EA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EBAY   = ["EBAY",   "EBAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ECG    = ["ECG",    "ECG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ECL    = ["ECL",    "ECL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ED     = ["ED",     "ED",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EEFT   = ["EEFT",   "EEFT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EFX    = ["EFX",    "EFX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EG     = ["EG",     "EG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EGP    = ["EGP",    "EGP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EHC    = ["EHC",    "EHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EIX    = ["EIX",    "EIX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EL     = ["EL",     "EL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ELAN   = ["ELAN",   "ELAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ELF    = ["ELF",    "ELF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ELS    = ["ELS",    "ELS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ELV    = ["ELV",    "ELV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EME    = ["EME",    "EME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EMN    = ["EMN",    "EMN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EMR    = ["EMR",    "EMR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ENPH   = ["ENPH",   "ENPH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ENTG   = ["ENTG",   "ENTG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EOG    = ["EOG",    "EOG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EPAM   = ["EPAM",   "EPAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EPR    = ["EPR",    "EPR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EQH    = ["EQH",    "EQH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EQIX   = ["EQIX",   "EQIX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EQR    = ["EQR",    "EQR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EQT    = ["EQT",    "EQT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ES     = ["ES",     "ES",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ESAB   = ["ESAB",   "ESAB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ESI    = ["ESI",    "ESI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ESS    = ["ESS",    "ESS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ESTC   = ["ESTC",   "ESTC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ETN    = ["ETN",    "ETN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ETR    = ["ETR",    "ETR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ETSY   = ["ETSY",   "ETSY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EVR    = ["EVR",    "EVR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EVRG   = ["EVRG",   "EVRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EW     = ["EW",     "EW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EWBC   = ["EWBC",   "EWBC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EXC    = ["EXC",    "EXC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EXE    = ["EXE",    "EXE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EXEL   = ["EXEL",   "EXEL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EXLS   = ["EXLS",   "EXLS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EXP    = ["EXP",    "EXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EXPD   = ["EXPD",   "EXPD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EXPE   = ["EXPE",   "EXPE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EXR    = ["EXR",    "EXR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    F      = ["F",      "F",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FAF    = ["FAF",    "FAF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FANG   = ["FANG",   "FANG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FAST   = ["FAST",   "FAST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FBIN   = ["FBIN",   "FBIN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FCN    = ["FCN",    "FCN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FCNCA  = ["FCNCA",  "FCNCA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FCX    = ["FCX",    "FCX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FDS    = ["FDS",    "FDS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FDX    = ["FDX",    "FDX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FDXF   = ["FDXF",   "FDXF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FE     = ["FE",     "FE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FERG   = ["FERG",   "FERG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FFIV   = ["FFIV",   "FFIV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FHB    = ["FHB",    "FHB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FHN    = ["FHN",    "FHN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FICO   = ["FICO",   "FICO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FIGR   = ["FIGR",   "FIGR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FIS    = ["FIS",    "FIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FISV   = ["FISV",   "FISV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FITB   = ["FITB",   "FITB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FIVE   = ["FIVE",   "FIVE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FIX    = ["FIX",    "FIX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FLEX   = ["FLEX",   "FLEX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FLO    = ["FLO",    "FLO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FLS    = ["FLS",    "FLS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FLUT   = ["FLUT",   "FLUT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FMC    = ["FMC",    "FMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FNB    = ["FNB",    "FNB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FND    = ["FND",    "FND",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FNF    = ["FNF",    "FNF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FOUR   = ["FOUR",   "FOUR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FOX    = ["FOX",    "FOX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FOXA   = ["FOXA",   "FOXA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FR     = ["FR",     "FR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FRHC   = ["FRHC",   "FRHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FRMI   = ["FRMI",   "FRMI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FRPT   = ["FRPT",   "FRPT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FRT    = ["FRT",    "FRT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FSLR   = ["FSLR",   "FSLR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FTAI   = ["FTAI",   "FTAI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FTI    = ["FTI",    "FTI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FTNT   = ["FTNT",   "FTNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FTV    = ["FTV",    "FTV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FWONA  = ["FWONA",  "FWONA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    FWONK  = ["FWONK",  "FWONK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    G      = ["G",      "G",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GAP    = ["GAP",    "GAP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GD     = ["GD",     "GD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GDDY   = ["GDDY",   "GDDY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GE     = ["GE",     "GE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GEHC   = ["GEHC",   "GEHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GEN    = ["GEN",    "GEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GEV    = ["GEV",    "GEV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GFS    = ["GFS",    "GFS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GGG    = ["GGG",    "GGG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GILD   = ["GILD",   "GILD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GIS    = ["GIS",    "GIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GL     = ["GL",     "GL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GLIBA  = ["GLIBA",  "GLIBA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GLIBK  = ["GLIBK",  "GLIBK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GLOB   = ["GLOB",   "GLOB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GLPI   = ["GLPI",   "GLPI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GLW    = ["GLW",    "GLW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GM     = ["GM",     "GM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GME    = ["GME",    "GME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GMED   = ["GMED",   "GMED",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GNRC   = ["GNRC",   "GNRC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GNTX   = ["GNTX",   "GNTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GOOG   = ["GOOG",   "GOOG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GPC    = ["GPC",    "GPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GPK    = ["GPK",    "GPK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GPN    = ["GPN",    "GPN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GRMN   = ["GRMN",   "GRMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GS     = ["GS",     "GS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GTES   = ["GTES",   "GTES",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GTLB   = ["GTLB",   "GTLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GTM    = ["GTM",    "GTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GWRE   = ["GWRE",   "GWRE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GWW    = ["GWW",    "GWW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    GXO    = ["GXO",    "GXO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    H      = ["H",      "H",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HAL    = ["HAL",    "HAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HALO   = ["HALO",   "HALO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HAS    = ["HAS",    "HAS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HAYW   = ["HAYW",   "HAYW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HBAN   = ["HBAN",   "HBAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HCA    = ["HCA",    "HCA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HEI    = ["HEI",    "HEI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HEI_A  = ["HEI_A",  "HEI.A",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HHH    = ["HHH",    "HHH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HIG    = ["HIG",    "HIG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HII    = ["HII",    "HII",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HIW    = ["HIW",    "HIW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HLI    = ["HLI",    "HLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HLNE   = ["HLNE",   "HLNE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HLT    = ["HLT",    "HLT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HOG    = ["HOG",    "HOG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HON    = ["HON",    "HON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HPE    = ["HPE",    "HPE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HPQ    = ["HPQ",    "HPQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HR     = ["HR",     "HR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HRB    = ["HRB",    "HRB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HRL    = ["HRL",    "HRL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HSIC   = ["HSIC",   "HSIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HST    = ["HST",    "HST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HSY    = ["HSY",    "HSY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HUBB   = ["HUBB",   "HUBB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HUBS   = ["HUBS",   "HUBS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HUM    = ["HUM",    "HUM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HUN    = ["HUN",    "HUN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HWM    = ["HWM",    "HWM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    HXL    = ["HXL",    "HXL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IBKR   = ["IBKR",   "IBKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IBM    = ["IBM",    "IBM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ICE    = ["ICE",    "ICE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IDA    = ["IDA",    "IDA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IDXX   = ["IDXX",   "IDXX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IEX    = ["IEX",    "IEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IFF    = ["IFF",    "IFF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ILMN   = ["ILMN",   "ILMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INCY   = ["INCY",   "INCY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INGM   = ["INGM",   "INGM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INGR   = ["INGR",   "INGR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INSM   = ["INSM",   "INSM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INSP   = ["INSP",   "INSP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INTU   = ["INTU",   "INTU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INVH   = ["INVH",   "INVH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IONS   = ["IONS",   "IONS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IOT    = ["IOT",    "IOT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IP     = ["IP",     "IP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IPGP   = ["IPGP",   "IPGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IQV    = ["IQV",    "IQV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IR     = ["IR",     "IR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IRDM   = ["IRDM",   "IRDM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IRM    = ["IRM",    "IRM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ISRG   = ["ISRG",   "ISRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IT     = ["IT",     "IT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ITT    = ["ITT",    "ITT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ITW    = ["ITW",    "ITW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IVZ    = ["IVZ",    "IVZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    J      = ["J",      "J",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JAZZ   = ["JAZZ",   "JAZZ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JBHT   = ["JBHT",   "JBHT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JBL    = ["JBL",    "JBL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JCI    = ["JCI",    "JCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JEF    = ["JEF",    "JEF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JHX    = ["JHX",    "JHX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JKHY   = ["JKHY",   "JKHY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JLL    = ["JLL",    "JLL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    JNJ    = ["JNJ",    "JNJ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KBR    = ["KBR",    "KBR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KD     = ["KD",     "KD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KDP    = ["KDP",    "KDP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KEX    = ["KEX",    "KEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KEY    = ["KEY",    "KEY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KEYS   = ["KEYS",   "KEYS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KHC    = ["KHC",    "KHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KIM    = ["KIM",    "KIM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KKR    = ["KKR",    "KKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KLAC   = ["KLAC",   "KLAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KMB    = ["KMB",    "KMB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KMI    = ["KMI",    "KMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KMPR   = ["KMPR",   "KMPR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KMX    = ["KMX",    "KMX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KNSL   = ["KNSL",   "KNSL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KNX    = ["KNX",    "KNX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KO     = ["KO",     "KO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KR     = ["KR",     "KR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KRC    = ["KRC",    "KRC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KRMN   = ["KRMN",   "KRMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    KVUE   = ["KVUE",   "KVUE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    L      = ["L",      "L",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LAD    = ["LAD",    "LAD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LAMR   = ["LAMR",   "LAMR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LAZ    = ["LAZ",    "LAZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LBRDA  = ["LBRDA",  "LBRDA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LBRDK  = ["LBRDK",  "LBRDK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LBTYA  = ["LBTYA",  "LBTYA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LBTYK  = ["LBTYK",  "LBTYK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LCID   = ["LCID",   "LCID",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LDOS   = ["LDOS",   "LDOS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LEA    = ["LEA",    "LEA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LECO   = ["LECO",   "LECO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LEN    = ["LEN",    "LEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LEN_B  = ["LEN_B",  "LEN.B",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LFUS   = ["LFUS",   "LFUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LH     = ["LH",     "LH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LHX    = ["LHX",    "LHX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LII    = ["LII",    "LII",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LIN    = ["LIN",    "LIN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LINE   = ["LINE",   "LINE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LITE   = ["LITE",   "LITE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LKQ    = ["LKQ",    "LKQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LLY    = ["LLY",    "LLY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LLYVA  = ["LLYVA",  "LLYVA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LLYVK  = ["LLYVK",  "LLYVK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LMT    = ["LMT",    "LMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LNC    = ["LNC",    "LNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LNG    = ["LNG",    "LNG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LNT    = ["LNT",    "LNT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LOAR   = ["LOAR",   "LOAR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LOPE   = ["LOPE",   "LOPE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LOW    = ["LOW",    "LOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LPLA   = ["LPLA",   "LPLA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LPX    = ["LPX",    "LPX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LRCX   = ["LRCX",   "LRCX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LSCC   = ["LSCC",   "LSCC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LSTR   = ["LSTR",   "LSTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LULU   = ["LULU",   "LULU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LUV    = ["LUV",    "LUV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LVS    = ["LVS",    "LVS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LW     = ["LW",     "LW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LYB    = ["LYB",    "LYB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LYFT   = ["LYFT",   "LYFT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    LYV    = ["LYV",    "LYV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    M      = ["M",      "M",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MAA    = ["MAA",    "MAA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MAN    = ["MAN",    "MAN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MANH   = ["MANH",   "MANH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MAR    = ["MAR",    "MAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MAS    = ["MAS",    "MAS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MAT    = ["MAT",    "MAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MCD    = ["MCD",    "MCD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MCHP   = ["MCHP",   "MCHP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MCK    = ["MCK",    "MCK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MCO    = ["MCO",    "MCO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MDB    = ["MDB",    "MDB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MDLN   = ["MDLN",   "MDLN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MDLZ   = ["MDLZ",   "MDLZ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MDT    = ["MDT",    "MDT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MDU    = ["MDU",    "MDU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MEDP   = ["MEDP",   "MEDP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MET    = ["MET",    "MET",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MGM    = ["MGM",    "MGM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MHK    = ["MHK",    "MHK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MIDD   = ["MIDD",   "MIDD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MKC    = ["MKC",    "MKC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MKL    = ["MKL",    "MKL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MKSI   = ["MKSI",   "MKSI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MKTX   = ["MKTX",   "MKTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MLI    = ["MLI",    "MLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MLM    = ["MLM",    "MLM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MMM    = ["MMM",    "MMM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MNST   = ["MNST",   "MNST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MO     = ["MO",     "MO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MOH    = ["MOH",    "MOH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MORN   = ["MORN",   "MORN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MOS    = ["MOS",    "MOS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MP     = ["MP",     "MP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MPC    = ["MPC",    "MPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MPT    = ["MPT",    "MPT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MPWR   = ["MPWR",   "MPWR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MRK    = ["MRK",    "MRK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MRNA   = ["MRNA",   "MRNA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MRP    = ["MRP",    "MRP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MRSH   = ["MRSH",   "MRSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MRVL   = ["MRVL",   "MRVL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MS     = ["MS",     "MS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MSA    = ["MSA",    "MSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MSCI   = ["MSCI",   "MSCI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MSGS   = ["MSGS",   "MSGS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MSI    = ["MSI",    "MSI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MSM    = ["MSM",    "MSM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MTB    = ["MTB",    "MTB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MTCH   = ["MTCH",   "MTCH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MTD    = ["MTD",    "MTD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MTDR   = ["MTDR",   "MTDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MTG    = ["MTG",    "MTG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MTN    = ["MTN",    "MTN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MTSI   = ["MTSI",   "MTSI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MTZ    = ["MTZ",    "MTZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    MUSA   = ["MUSA",   "MUSA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NBIX   = ["NBIX",   "NBIX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NCLH   = ["NCLH",   "NCLH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NCNO   = ["NCNO",   "NCNO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NDAQ   = ["NDAQ",   "NDAQ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NDSN   = ["NDSN",   "NDSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NEE    = ["NEE",    "NEE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NEM    = ["NEM",    "NEM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NET    = ["NET",    "NET",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NEU    = ["NEU",    "NEU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NFG    = ["NFG",    "NFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NI     = ["NI",     "NI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NIQ    = ["NIQ",    "NIQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NKE    = ["NKE",    "NKE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NLY    = ["NLY",    "NLY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NNN    = ["NNN",    "NNN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NOC    = ["NOC",    "NOC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NOV    = ["NOV",    "NOV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NOW    = ["NOW",    "NOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NRG    = ["NRG",    "NRG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NSA    = ["NSA",    "NSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NSC    = ["NSC",    "NSC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NTAP   = ["NTAP",   "NTAP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NTNX   = ["NTNX",   "NTNX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NTRA   = ["NTRA",   "NTRA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NTRS   = ["NTRS",   "NTRS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NU     = ["NU",     "NU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NUE    = ["NUE",    "NUE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NVR    = ["NVR",    "NVR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NVST   = ["NVST",   "NVST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NVT    = ["NVT",    "NVT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NWL    = ["NWL",    "NWL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NWS    = ["NWS",    "NWS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NWSA   = ["NWSA",   "NWSA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NXST   = ["NXST",   "NXST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    NYT    = ["NYT",    "NYT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    O      = ["O",      "O",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OC     = ["OC",     "OC",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ODFL   = ["ODFL",   "ODFL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OGE    = ["OGE",    "OGE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OGN    = ["OGN",    "OGN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OHI    = ["OHI",    "OHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OKE    = ["OKE",    "OKE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OKTA   = ["OKTA",   "OKTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OLED   = ["OLED",   "OLED",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OLLI   = ["OLLI",   "OLLI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OLN    = ["OLN",    "OLN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OMC    = ["OMC",    "OMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OMF    = ["OMF",    "OMF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ON     = ["ON",     "ON",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ONON   = ["ONON",   "ONON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ONTO   = ["ONTO",   "ONTO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ORI    = ["ORI",    "ORI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ORLY   = ["ORLY",   "ORLY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OSK    = ["OSK",    "OSK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OTIS   = ["OTIS",   "OTIS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OVV    = ["OVV",    "OVV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OWL    = ["OWL",    "OWL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OXY    = ["OXY",    "OXY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    OZK    = ["OZK",    "OZK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    P      = ["P",      "P",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PAG    = ["PAG",    "PAG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PANW   = ["PANW",   "PANW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PATH   = ["PATH",   "PATH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PAYC   = ["PAYC",   "PAYC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PAYX   = ["PAYX",   "PAYX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PB     = ["PB",     "PB",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PCAR   = ["PCAR",   "PCAR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PCG    = ["PCG",    "PCG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PCOR   = ["PCOR",   "PCOR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PCTY   = ["PCTY",   "PCTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PEG    = ["PEG",    "PEG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PEGA   = ["PEGA",   "PEGA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PEN    = ["PEN",    "PEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PENN   = ["PENN",   "PENN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PEP    = ["PEP",    "PEP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PFE    = ["PFE",    "PFE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PFG    = ["PFG",    "PFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PFGC   = ["PFGC",   "PFGC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PG     = ["PG",     "PG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PGR    = ["PGR",    "PGR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PH     = ["PH",     "PH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PHM    = ["PHM",    "PHM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PINS   = ["PINS",   "PINS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PK     = ["PK",     "PK",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PKG    = ["PKG",    "PKG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PLD    = ["PLD",    "PLD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PLNT   = ["PLNT",   "PLNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PM     = ["PM",     "PM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PNC    = ["PNC",    "PNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PNFP   = ["PNFP",   "PNFP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PNR    = ["PNR",    "PNR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PNW    = ["PNW",    "PNW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PODD   = ["PODD",   "PODD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    POOL   = ["POOL",   "POOL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    POST   = ["POST",   "POST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PPC    = ["PPC",    "PPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PPG    = ["PPG",    "PPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PPL    = ["PPL",    "PPL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PPLI   = ["PPLI",   "PPLI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PR     = ["PR",     "PR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PRGO   = ["PRGO",   "PRGO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PRI    = ["PRI",    "PRI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PRMB   = ["PRMB",   "PRMB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PRU    = ["PRU",    "PRU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PSA    = ["PSA",    "PSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PSN    = ["PSN",    "PSN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PSX    = ["PSX",    "PSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PTC    = ["PTC",    "PTC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PVH    = ["PVH",    "PVH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PWR    = ["PWR",    "PWR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    PYPL   = ["PYPL",   "PYPL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    Q      = ["Q",      "Q",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    QCOM   = ["QCOM",   "QCOM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    QGEN   = ["QGEN",   "QGEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    QRVO   = ["QRVO",   "QRVO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    QS     = ["QS",     "QS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    QSR    = ["QSR",    "QSR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    QXO    = ["QXO",    "QXO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    R      = ["R",      "R",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RAL    = ["RAL",    "RAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RARE   = ["RARE",   "RARE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RBA    = ["RBA",    "RBA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RBC    = ["RBC",    "RBC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RBLX   = ["RBLX",   "RBLX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RBRK   = ["RBRK",   "RBRK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RCL    = ["RCL",    "RCL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RDDT   = ["RDDT",   "RDDT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    REG    = ["REG",    "REG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    REGN   = ["REGN",   "REGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    REXR   = ["REXR",   "REXR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    REYN   = ["REYN",   "REYN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RF     = ["RF",     "RF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RGA    = ["RGA",    "RGA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RGEN   = ["RGEN",   "RGEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RGLD   = ["RGLD",   "RGLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RH     = ["RH",     "RH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RHI    = ["RHI",    "RHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RITM   = ["RITM",   "RITM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RIVN   = ["RIVN",   "RIVN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RJF    = ["RJF",    "RJF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RKLB   = ["RKLB",   "RKLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RKT    = ["RKT",    "RKT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RL     = ["RL",     "RL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RLI    = ["RLI",    "RLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RMD    = ["RMD",    "RMD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RNG    = ["RNG",    "RNG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RNR    = ["RNR",    "RNR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ROIV   = ["ROIV",   "ROIV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ROK    = ["ROK",    "ROK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ROKU   = ["ROKU",   "ROKU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ROL    = ["ROL",    "ROL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ROP    = ["ROP",    "ROP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ROST   = ["ROST",   "ROST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RPM    = ["RPM",    "RPM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RPRX   = ["RPRX",   "RPRX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RRC    = ["RRC",    "RRC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RRX    = ["RRX",    "RRX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RS     = ["RS",     "RS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RSG    = ["RSG",    "RSG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RTX    = ["RTX",    "RTX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RVMD   = ["RVMD",   "RVMD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RVTY   = ["RVTY",   "RVTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RYAN   = ["RYAN",   "RYAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    RYN    = ["RYN",    "RYN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    S      = ["S",      "S",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SAIA   = ["SAIA",   "SAIA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SAIC   = ["SAIC",   "SAIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SAIL   = ["SAIL",   "SAIL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SAM    = ["SAM",    "SAM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SARO   = ["SARO",   "SARO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SBAC   = ["SBAC",   "SBAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SCCO   = ["SCCO",   "SCCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SCHW   = ["SCHW",   "SCHW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SCI    = ["SCI",    "SCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SEB    = ["SEB",    "SEB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SEIC   = ["SEIC",   "SEIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SF     = ["SF",     "SF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SFD    = ["SFD",    "SFD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SFM    = ["SFM",    "SFM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SGI    = ["SGI",    "SGI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SHC    = ["SHC",    "SHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SHW    = ["SHW",    "SHW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SIRI   = ["SIRI",   "SIRI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SITE   = ["SITE",   "SITE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SJM    = ["SJM",    "SJM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SLB    = ["SLB",    "SLB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SLGN   = ["SLGN",   "SLGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SLM    = ["SLM",    "SLM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SMCI   = ["SMCI",   "SMCI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SMG    = ["SMG",    "SMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SMMT   = ["SMMT",   "SMMT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SN     = ["SN",     "SN",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SNA    = ["SNA",    "SNA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SNDR   = ["SNDR",   "SNDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SNOW   = ["SNOW",   "SNOW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SNPS   = ["SNPS",   "SNPS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SNX    = ["SNX",    "SNX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SO     = ["SO",     "SO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SOFI   = ["SOFI",   "SOFI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SOLS   = ["SOLS",   "SOLS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SOLV   = ["SOLV",   "SOLV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SON    = ["SON",    "SON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SPG    = ["SPG",    "SPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SPGI   = ["SPGI",   "SPGI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SPOT   = ["SPOT",   "SPOT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SRE    = ["SRE",    "SRE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SRPT   = ["SRPT",   "SRPT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SSB    = ["SSB",    "SSB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SSD    = ["SSD",    "SSD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SSNC   = ["SSNC",   "SSNC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ST     = ["ST",     "ST",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    STAG   = ["STAG",   "STAG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    STE    = ["STE",    "STE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    STLD   = ["STLD",   "STLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    STT    = ["STT",    "STT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    STWD   = ["STWD",   "STWD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    STZ    = ["STZ",    "STZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SUI    = ["SUI",    "SUI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SW     = ["SW",     "SW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SWK    = ["SWK",    "SWK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SWKS   = ["SWKS",   "SWKS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SYF    = ["SYF",    "SYF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SYK    = ["SYK",    "SYK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    SYY    = ["SYY",    "SYY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TAP    = ["TAP",    "TAP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TDC    = ["TDC",    "TDC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TDG    = ["TDG",    "TDG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TDY    = ["TDY",    "TDY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TEAM   = ["TEAM",   "TEAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TECH   = ["TECH",   "TECH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TEM    = ["TEM",    "TEM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TER    = ["TER",    "TER",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TFC    = ["TFC",    "TFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TFSL   = ["TFSL",   "TFSL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TFX    = ["TFX",    "TFX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TGT    = ["TGT",    "TGT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    THC    = ["THC",    "THC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    THG    = ["THG",    "THG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    THO    = ["THO",    "THO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TIGO   = ["TIGO",   "TIGO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TJX    = ["TJX",    "TJX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TKO    = ["TKO",    "TKO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TKR    = ["TKR",    "TKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TLN    = ["TLN",    "TLN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TMO    = ["TMO",    "TMO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TMUS   = ["TMUS",   "TMUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TNL    = ["TNL",    "TNL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TOL    = ["TOL",    "TOL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TOST   = ["TOST",   "TOST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TPG    = ["TPG",    "TPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TPL    = ["TPL",    "TPL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TPR    = ["TPR",    "TPR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TREX   = ["TREX",   "TREX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TRGP   = ["TRGP",   "TRGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TRMB   = ["TRMB",   "TRMB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TROW   = ["TROW",   "TROW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TRU    = ["TRU",    "TRU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TRV    = ["TRV",    "TRV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TSCO   = ["TSCO",   "TSCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TSN    = ["TSN",    "TSN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TT     = ["TT",     "TT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TTC    = ["TTC",    "TTC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TTD    = ["TTD",    "TTD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TTEK   = ["TTEK",   "TTEK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TTWO   = ["TTWO",   "TTWO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TW     = ["TW",     "TW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TWLO   = ["TWLO",   "TWLO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TXN    = ["TXN",    "TXN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TXRH   = ["TXRH",   "TXRH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TXT    = ["TXT",    "TXT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    TYL    = ["TYL",    "TYL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    U      = ["U",      "U",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UA     = ["UA",     "UA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UAA    = ["UAA",    "UAA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UAL    = ["UAL",    "UAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UDR    = ["UDR",    "UDR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UGI    = ["UGI",    "UGI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UHAL   = ["UHAL",   "UHAL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UHAL_B = ["UHAL_B", "UHAL.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UHS    = ["UHS",    "UHS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UI     = ["UI",     "UI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ULTA   = ["ULTA",   "ULTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UNH    = ["UNH",    "UNH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UNM    = ["UNM",    "UNM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UNP    = ["UNP",    "UNP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UPS    = ["UPS",    "UPS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    URI    = ["URI",    "URI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    USB    = ["USB",    "USB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    USFD   = ["USFD",   "USFD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UTHR   = ["UTHR",   "UTHR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    UWMC   = ["UWMC",   "UWMC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VEEV   = ["VEEV",   "VEEV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VFC    = ["VFC",    "VFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VGNT   = ["VGNT",   "VGNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VICI   = ["VICI",   "VICI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VIK    = ["VIK",    "VIK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VIRT   = ["VIRT",   "VIRT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VKTX   = ["VKTX",   "VKTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VLO    = ["VLO",    "VLO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VLTO   = ["VLTO",   "VLTO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VMC    = ["VMC",    "VMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VMI    = ["VMI",    "VMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VNO    = ["VNO",    "VNO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VNOM   = ["VNOM",   "VNOM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VNT    = ["VNT",    "VNT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VOYA   = ["VOYA",   "VOYA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VRSK   = ["VRSK",   "VRSK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VRSN   = ["VRSN",   "VRSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VRT    = ["VRT",    "VRT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VRTX   = ["VRTX",   "VRTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VSNT   = ["VSNT",   "VSNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VST    = ["VST",    "VST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VTR    = ["VTR",    "VTR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VTRS   = ["VTRS",   "VTRS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VVV    = ["VVV",    "VVV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VZ     = ["VZ",     "VZ",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    W      = ["W",      "W",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WAB    = ["WAB",    "WAB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WAL    = ["WAL",    "WAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WAT    = ["WAT",    "WAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WBD    = ["WBD",    "WBD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WBS    = ["WBS",    "WBS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WCC    = ["WCC",    "WCC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WDAY   = ["WDAY",   "WDAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WDC    = ["WDC",    "WDC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WEC    = ["WEC",    "WEC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WELL   = ["WELL",   "WELL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WEN    = ["WEN",    "WEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WEX    = ["WEX",    "WEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WFC    = ["WFC",    "WFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WFRD   = ["WFRD",   "WFRD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WH     = ["WH",     "WH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WHR    = ["WHR",    "WHR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WING   = ["WING",   "WING",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WLK    = ["WLK",    "WLK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WM     = ["WM",     "WM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WMB    = ["WMB",    "WMB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WMS    = ["WMS",    "WMS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WMT    = ["WMT",    "WMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WPC    = ["WPC",    "WPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WRB    = ["WRB",    "WRB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WSC    = ["WSC",    "WSC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WSM    = ["WSM",    "WSM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WSO    = ["WSO",    "WSO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WST    = ["WST",    "WST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WTFC   = ["WTFC",   "WTFC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WTM    = ["WTM",    "WTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WTRG   = ["WTRG",   "WTRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WTW    = ["WTW",    "WTW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WU     = ["WU",     "WU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WWD    = ["WWD",    "WWD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WY     = ["WY",     "WY",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    WYNN   = ["WYNN",   "WYNN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XEL    = ["XEL",    "XEL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XOM    = ["XOM",    "XOM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XP     = ["XP",     "XP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XPO    = ["XPO",    "XPO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XRAY   = ["XRAY",   "XRAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XYL    = ["XYL",    "XYL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XYZ    = ["XYZ",    "XYZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    YETI   = ["YETI",   "YETI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    YUM    = ["YUM",    "YUM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    Z      = ["Z",      "Z",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ZBH    = ["ZBH",    "ZBH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ZBRA   = ["ZBRA",   "ZBRA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ZG     = ["ZG",     "ZG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ZION   = ["ZION",   "ZION",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ZM     = ["ZM",     "ZM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ZS     = ["ZS",     "ZS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    ZTS    = ["ZTS",    "ZTS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]

    # Equities - Sector ETFs (22)
    XLK  = ["XLK",  "XLK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VGT  = ["VGT",  "VGT",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLF  = ["XLF",  "XLF",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VFH  = ["VFH",  "VFH",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLY  = ["XLY",  "XLY",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VCR  = ["VCR",  "VCR",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLC  = ["XLC",  "XLC",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VOX  = ["VOX",  "VOX",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLV  = ["XLV",  "XLV",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VHT  = ["VHT",  "VHT",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLI  = ["XLI",  "XLI",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VIS  = ["VIS",  "VIS",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLP  = ["XLP",  "XLP",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VDC  = ["VDC",  "VDC",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLE  = ["XLE",  "XLE",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VDE  = ["VDE",  "VDE",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLB  = ["XLB",  "XLB",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VAW  = ["VAW",  "VAW",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLU  = ["XLU",  "XLU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VPU  = ["VPU",  "VPU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    XLRE = ["XLRE", "XLRE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VNQ  = ["VNQ",  "VNQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]

    # Index ETFs (broad market & international)
    SPY  = ["SPY",  "SPY",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    QQQ  = ["QQQ",  "QQQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    DIA  = ["DIA",  "DIA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IWM  = ["IWM",  "IWM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EWU  = ["EWU",  "EWU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EWG  = ["EWG",  "EWG",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EWJ  = ["EWJ",  "EWJ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EWH  = ["EWH",  "EWH",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EWA  = ["EWA",  "EWA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EWQ  = ["EWQ",  "EWQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    EFA  = ["EFA",  "EFA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    IEMG = ["IEMG", "IEMG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    INDA = ["INDA", "INDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]
    VT   = ["VT",   "VT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(1.0)]

    # indices (no longer allowed for trading as we moved to equities tickers instead)
    SPX   = ["SPX",   "SPX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    DJI   = ["DJI",   "DJI",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    NDX   = ["NDX",   "NDX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    VIX   = ["VIX",   "VIX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    FTSE  = ["FTSE",  "FTSE",  0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GDAXI = ["GDAXI", "GDAXI", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]

    # Hyperliquid Trade Pairs (USDC-quoted, src=HYPERLIQUID)
    # Crypto perp futures
    BTCUSDC   = ["BTCUSDC",   "BTC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(5.0)]
    ETHUSDC   = ["ETHUSDC",   "ETH/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(5.0)]
    SOLUSDC   = ["SOLUSDC",   "SOL/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    BNBUSDC   = ["BNBUSDC",   "BNB/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    XRPUSDC   = ["XRPUSDC",   "XRP/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    DOGEUSDC  = ["DOGEUSDC",  "DOGE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    ADAUSDC   = ["ADAUSDC",   "ADA/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    AVAXUSDC  = ["AVAXUSDC",  "AVAX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    LINKUSDC  = ["LINKUSDC",  "LINK/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    DOTUSDC   = ["DOTUSDC",   "DOT/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    TONUSDC   = ["TONUSDC",   "TON/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TRXUSDC   = ["TRXUSDC",   "TRX/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    LTCUSDC   = ["LTCUSDC",   "LTC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    TAOUSDC   = ["TAOUSDC",   "TAO/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    SUIUSDC   = ["SUIUSDC",   "SUI/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    ARBUSDC   = ["ARBUSDC",   "ARB/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    NEARUSDC  = ["NEARUSDC",  "NEAR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    ALGOUSDC  = ["ALGOUSDC",  "ALGO/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    ASTERUSDC = ["ASTERUSDC", "ASTER/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    UNIUSDC   = ["UNIUSDC",   "UNI/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    AAVEUSDC  = ["AAVEUSDC",  "AAVE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    CRVUSDC   = ["CRVUSDC",   "CRV/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    HYPEUSDC  = ["HYPEUSDC",  "HYPE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    XMRUSDC   = ["XMRUSDC",   "XMR/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    ZECUSDC   = ["ZECUSDC",   "ZEC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    PAXGUSDC  = ["PAXGUSDC",  "PAXG/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ENAUSDC   = ["ENAUSDC",   "ENA/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    ZROUSDC   = ["ZROUSDC",   "ZRO/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    WLDUSDC   = ["WLDUSDC",   "WLD/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    PUMPUSDC  = ["PUMPUSDC",  "PUMP/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]
    KPEPEUSDC = ["kPEPEUSDC", "kPEPE/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(2.0)]

    # Commodity perp futures (synthetic, track commodity prices — not physical delivery)
    WTIOILUSDC   = ["WTIOILUSDC",   "WTIOIL/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:CL",       InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    BRENTOILUSDC = ["BRENTOILUSDC", "BRENTOIL/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:BRENTOIL", InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    GOLDUSDC     = ["GOLDUSDC",     "GOLD/USDC",     0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:GOLD",     InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    SILVERUSDC   = ["SILVERUSDC",   "SILVER/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:SILVER",   InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    COPPERUSDC   = ["COPPERUSDC",   "COPPER/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:COPPER",   InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    NATGASUSDC   = ["NATGASUSDC",   "NATGAS/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:NATGAS",   InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]
    PLATINUMUSDC = ["PLATINUMUSDC", "PLATINUM/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:PLATINUM", InstrumentType.PERP, SubaccountTierBaseLeverage(3.0)]

    # Index perp futures (synthetic, track equity index prices — not ETFs)
    SP500USDC  = ["SP500USDC",  "SP500/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:SP500",  InstrumentType.PERP, SubaccountTierBaseLeverage(10.0)]
    XYZ100USDC = ["XYZ100USDC", "XYZ100/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:XYZ100", InstrumentType.PERP, SubaccountTierBaseLeverage(10.0)]
    EWYUSDC    = ["EWYUSDC",    "EWY/USDC",    0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:EWY",    InstrumentType.PERP, SubaccountTierBaseLeverage(5.0)]

    # Equity perp futures (synthetic, track single-stock prices — not actual shares)
    NVDAUSDC  = ["NVDAUSDC",  "NVDA/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:NVDA",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    AAPLUSDC  = ["AAPLUSDC",  "AAPL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AAPL",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    TSLAUSDC  = ["TSLAUSDC",  "TSLA/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:TSLA",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    MSFTUSDC  = ["MSFTUSDC",  "MSFT/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MSFT",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    AMZNUSDC  = ["AMZNUSDC",  "AMZN/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AMZN",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    GOOGLUSDC = ["GOOGLUSDC", "GOOGL/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:GOOGL", InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    METAUSDC  = ["METAUSDC",  "META/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:META",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    COINUSDC  = ["COINUSDC",  "COIN/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:COIN",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    CRCLUSDC  = ["CRCLUSDC",  "CRCL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:CRCL",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    MSTRUSDC  = ["MSTRUSDC",  "MSTR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MSTR",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    PLTRUSDC  = ["PLTRUSDC",  "PLTR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:PLTR",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    AMDUSDC   = ["AMDUSDC",   "AMD/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AMD",   InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    TSMUSDC   = ["TSMUSDC",   "TSM/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:TSM",   InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    NFLXUSDC  = ["NFLXUSDC",  "NFLX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:NFLX",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    SNDKUSDC  = ["SNDKUSDC",  "SNDK/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:SNDK",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    INTCUSDC  = ["INTCUSDC",  "INTC/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:INTC",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    MUUSDC    = ["MUUSDC",    "MU/USDC",    0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MU",    InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    HOODUSDC  = ["HOODUSDC",  "HOOD/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:HOOD",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    ORCLUSDC  = ["ORCLUSDC",  "ORCL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:ORCL",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    SPCXUSDC  = ["SPCXUSDC",  "SPCX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:SPCX",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]

    @property
    def trade_pair_id(self):
        return self.value[0]

    @property
    def trade_pair(self):
        return self.value[1]

    @property
    def fees(self):
        return self.value[2]

    @property
    def min_leverage(self):
        return self.value[3]

    @property
    def max_leverage(self):
        return self.value[4]

    @property
    def trade_pair_category(self):
        return self.value[5]

    @property
    def subcategory(self):
        if len(self.value) > 6 and isinstance(self.value[6], TradePairSubcategory):
            return self.value[6]
        return None

    @property
    def src(self) -> TradePairSource:
        if len(self.value) > 7 and isinstance(self.value[7], TradePairSource):
            return self.value[7]
        return TradePairSource.VANTA

    @property
    def hl_coin(self) -> str:
        # type() is str (not isinstance) to exclude InstrumentType, which is a str subclass via str-Enum.
        if len(self.value) > 8 and type(self.value[8]) is str:
            return self.value[8]
        return self.base

    @property
    def instrument_type(self) -> InstrumentType:
        """SPOT or PERP. Located by type scan — robust to future fields added anywhere in the value list."""
        for v in self.value:
            if isinstance(v, InstrumentType):
                return v
        raise ValueError(f"TradePair {self.trade_pair_id} is missing instrument_type")

    @property
    def subaccount_tier_base_leverage(self) -> float:
        """Per-pair Tier-1 base for subaccount order-entry tier dispatch.
        See leverage_utils.get_tier_positional_leverage.
        """
        for v in self.value:
            if isinstance(v, SubaccountTierBaseLeverage):
                return v.value
        raise ValueError(f"TradePair {self.trade_pair_id} is missing subaccount_tier_base_leverage")

    def transaction_fee_rate(self, is_hl_taker: bool | None = True) -> float:
        """Maker rate only when the fill added liquidity; taker when it took or is unknown."""
        if self.src != TradePairSource.HYPERLIQUID:
            return TRANSACTION_FEE_RATE.get(self.trade_pair_category, 0)
        taker, maker = HL_FEE_BY_CATEGORY[self.trade_pair_category]
        return maker if is_hl_taker is False else taker

    @property
    def carry_fee_rate_per_interval(self) -> float:
        if self.src == TradePairSource.HYPERLIQUID:
            return 0
        return CARRY_FEE_RATE_PER_INTERVAL.get(self.trade_pair_category, 0)

    @property
    def is_crypto(self):
        return self.trade_pair_category == TradePairCategory.CRYPTO

    @property
    def is_forex(self):
        return self.trade_pair_category == TradePairCategory.FOREX

    @property
    def is_equities(self):
        return self.trade_pair_category == TradePairCategory.EQUITIES

    @property
    def is_indices(self):
        return self.trade_pair_category == TradePairCategory.INDICES

    @property
    def is_commodities(self):
        return self.trade_pair_category == TradePairCategory.COMMODITIES

    @property
    def is_blocked(self) -> bool:
        """Check if this trade pair is blocked from trading"""
        return self.trade_pair_id in BLOCKED_TRADE_PAIR_IDS

    @property
    def is_flat_only(self) -> bool:
        """Check if this trade pair only allows flat orders"""
        return self.trade_pair_id in FLAT_ONLY_TRADE_PAIR_IDS

    @property
    def lot_size(self):
        trade_pair_lot_size_override = {
            'XAUUSD': 100,
            'XAGUSD': 5_000,
        }
        if self.trade_pair_id in trade_pair_lot_size_override:
            return trade_pair_lot_size_override[self.trade_pair_id]
        trade_pair_lot_size = {TradePairCategory.CRYPTO: 1,
                               TradePairCategory.FOREX: 100_000,
                               TradePairCategory.INDICES: 1,
                               TradePairCategory.EQUITIES: 1,
                               TradePairCategory.COMMODITIES: 1}
        return trade_pair_lot_size[self.trade_pair_category]

    @property
    def base(self):
        return self.trade_pair.split("/")[0]

    @property
    def quote(self):
        parts = self.trade_pair.split("/")
        return parts[1] if len(parts) > 1 else "USD"

    @classmethod
    def categories(cls):
        return {tp.trade_pair_id: tp.trade_pair_category.value for tp in cls}

    @classmethod
    def subcategories(cls):
        # Eventually we'll want subcategories for each trade pair
        trade_pairs_by_subcategory = defaultdict(list)
        for tp in cls:
            if tp.subcategory is not None:
                trade_pairs_by_subcategory[tp.subcategory.value].append(tp.trade_pair_id)
        return trade_pairs_by_subcategory

    @staticmethod
    def to_dict():
        # Convert TradePair Enum to a dictionary
        return {
            member.name: {
                "trade_pair_id": member.trade_pair_id,
                "trade_pair": member.trade_pair,
                "fees": member.fees,
                "min_leverage": member.min_leverage,
                "max_leverage": member.max_leverage,
            }
            for member in TradePair
        }

    @staticmethod
    def to_enum(stream_id):
        m_map = {member.name: member for member in TradePair}
        return m_map[stream_id]

    @staticmethod
    def from_trade_pair_id(trade_pair_id: str):
        """
        Converts a trade_pair_id string into a TradePair object.

        Args:
            trade_pair_id (str): The ID of the trade pair to convert.

        Returns:
            TradePair | None: The corresponding trade pair object.
        """
        return TRADE_PAIR_ID_TO_TRADE_PAIR.get(trade_pair_id)

    def __json__(self):
        # Provide a dictionary representation for JSON serialization
        return {
            "trade_pair_id": self.trade_pair_id,
            "trade_pair": self.trade_pair,
            "fees": self.fees,
            "min_leverage": self.min_leverage,
            "max_leverage": self.max_leverage,
            "trade_pair_category": self.trade_pair_category,
        }

    def debug_dict(self):
        return {
            "trade_pair_id": self.trade_pair_id,
            "trade_pair": self.trade_pair,
            "fees": self.fees,
            "min_leverage": self.min_leverage,
            "max_leverage": self.max_leverage,
        }

    @staticmethod
    def get_latest_trade_pair_from_trade_pair_id(trade_pair_id):
        return TRADE_PAIR_ID_TO_TRADE_PAIR.get(trade_pair_id)

    @staticmethod
    def get_latest_tade_pair_from_trade_pair_str(trade_pair_str):
        return TRADE_PAIR_STR_TO_TRADE_PAIR.get(trade_pair_str)

    def __str__(self):
        return str(self.trade_pair_id)


TRADE_PAIR_ID_TO_TRADE_PAIR = {x.trade_pair_id: x for x in TradePair}
TRADE_PAIR_STR_TO_TRADE_PAIR = {x.trade_pair: x for x in TradePair}
HL_COIN_TO_TRADE_PAIR: dict[str, TradePair] = {
    tp.hl_coin: tp for tp in TradePair if tp.src == TradePairSource.HYPERLIQUID
}

# Maps native Vanta crypto TradePairs to their Hyperliquid (USDC-quoted) equivalents.
# BCHUSD has no corresponding HL pair and is omitted.
NATIVE_CRYPTO_TO_HL_TRADE_PAIR: dict[TradePair, TradePair] = {
    TradePair.BTCUSD:  TradePair.BTCUSDC,
    TradePair.ETHUSD:  TradePair.ETHUSDC,
    TradePair.SOLUSD:  TradePair.SOLUSDC,
    TradePair.XRPUSD:  TradePair.XRPUSDC,
    TradePair.DOGEUSD: TradePair.DOGEUSDC,
    TradePair.ADAUSD:  TradePair.ADAUSDC,
    TradePair.TAOUSD:  TradePair.TAOUSDC,
    TradePair.HYPEUSD: TradePair.HYPEUSDC,
    TradePair.ZECUSD:  TradePair.ZECUSDC,
    TradePair.LINKUSD: TradePair.LINKUSDC,
    TradePair.XMRUSD:  TradePair.XMRUSDC,
    TradePair.LTCUSD:  TradePair.LTCUSDC,
}
