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

class ExposureGroup(str, Enum):
    """Correlated-exposure group for equities.

    Every single stock and sector ETF belongs to exactly one group; broad-market and country
    ETFs (SPY, QQQ, EFA, VT, ...) belong to none. Used to cap net exposure stacked across
    correlated equity pairs — see leverage_utils.get_correlation_legs.

    Values are the sector labels russell1000.csv uses verbatim, so a CSV row maps straight onto
    a member.
    """
    INFORMATION_TECHNOLOGY = "Information Technology"
    FINANCIALS             = "Financials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    COMMUNICATION          = "Communication"
    HEALTH_CARE            = "Health Care"
    INDUSTRIALS            = "Industrials"
    CONSUMER_STAPLES       = "Consumer Staples"
    ENERGY                 = "Energy"
    MATERIALS              = "Materials"
    UTILITIES              = "Utilities"
    REAL_ESTATE            = "Real Estate"


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

# Pro account fee schedule. Currently mirrors the standard schedule above so pro accounts
# are priced identically until real values are set.
PRO_CARRY_FEE_RATE_PER_INTERVAL = dict(CARRY_FEE_RATE_PER_INTERVAL)
PRO_DAILY_STOCK_BORROW_RATE = DAILY_STOCK_BORROW_RATE
PRO_DAILY_MARGIN_INTEREST_RATE = DAILY_MARGIN_INTEREST_RATE

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
    BTCUSD  = ["BTCUSD",  "BTC/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.MAJORS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    ETHUSD  = ["ETHUSD",  "ETH/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.MAJORS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    SOLUSD  = ["SOLUSD",  "SOL/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    XRPUSD  = ["XRPUSD",  "XRP/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    DOGEUSD = ["DOGEUSD", "DOGE/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    ADAUSD  = ["ADAUSD",  "ADA/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    TAOUSD  = ["TAOUSD",  "TAO/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    HYPEUSD = ["HYPEUSD", "HYPE/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    ZECUSD  = ["ZECUSD",  "ZEC/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    BCHUSD  = ["BCHUSD",  "BCH/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    LINKUSD = ["LINKUSD", "LINK/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    XMRUSD  = ["XMRUSD",  "XMR/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    LTCUSD  = ["LTCUSD",  "LTC/USD",  0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS,   InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]

    # forex
    AUDCAD = ["AUDCAD", "AUD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    AUDCHF = ["AUDCHF", "AUD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    AUDUSD = ["AUDUSD", "AUD/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    AUDJPY = ["AUDJPY", "AUD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    AUDNZD = ["AUDNZD", "AUD/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    CADCHF = ["CADCHF", "CAD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    CADJPY = ["CADJPY", "CAD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    CHFJPY = ["CHFJPY", "CHF/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    EURAUD = ["EURAUD", "EUR/AUD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    EURCAD = ["EURCAD", "EUR/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    EURUSD = ["EURUSD", "EUR/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    EURCHF = ["EURCHF", "EUR/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    EURGBP = ["EURGBP", "EUR/GBP", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    EURJPY = ["EURJPY", "EUR/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    EURNZD = ["EURNZD", "EUR/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    NZDCAD = ["NZDCAD", "NZD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    NZDCHF = ["NZDCHF", "NZD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    NZDJPY = ["NZDJPY", "NZD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    NZDUSD = ["NZDUSD", "NZD/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    GBPAUD = ["GBPAUD", "GBP/AUD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    GBPCAD = ["GBPCAD", "GBP/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    GBPCHF = ["GBPCHF", "GBP/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    GBPJPY = ["GBPJPY", "GBP/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    GBPNZD = ["GBPNZD", "GBP/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    GBPUSD = ["GBPUSD", "GBP/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    USDCAD = ["USDCAD", "USD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    USDCHF = ["USDCHF", "USD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    USDJPY = ["USDJPY", "USD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), True]
    USDMXN = ["USDMXN", "USD/MXN", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]


    # "Commodities" (Bundle with Forex for now)
    XAUUSD = ["XAUUSD", "XAU/USD", 0.00007, COMMODITIES_MIN_LEVERAGE, COMMODITIES_MAX_LEVERAGE, TradePairCategory.FOREX, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]
    XAGUSD = ["XAGUSD", "XAG/USD", 0.00007, COMMODITIES_MIN_LEVERAGE, COMMODITIES_MAX_LEVERAGE, TradePairCategory.FOREX, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]

    # Equities - Stocks
    # Technology (10)
    NVDA = ["NVDA", "NVDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MSFT = ["MSFT", "MSFT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AAPL = ["AAPL", "AAPL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AVGO = ["AVGO", "AVGO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TSM  = ["TSM",  "TSM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ORCL = ["ORCL", "ORCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AMD  = ["AMD",  "AMD",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MU   = ["MU",   "MU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    CRM  = ["CRM",  "CRM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    UBER = ["UBER", "UBER", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    # Financial Services (5)
    BRK_B = ["BRK_B", "BRK.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    JPM   = ["JPM",   "JPM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    V     = ["V",     "V",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    MA    = ["MA",    "MA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    BAC   = ["BAC",   "BAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    # Consumer Discretionary (5)
    AMZN = ["AMZN", "AMZN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    TSLA = ["TSLA", "TSLA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    HD   = ["HD",   "HD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    BABA = ["BABA", "BABA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    SBUX = ["SBUX", "SBUX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    # Communication Services (5)
    GOOGL = ["GOOGL", "GOOGL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    META  = ["META",  "META",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    NFLX  = ["NFLX",  "NFLX",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    APP   = ["APP",   "APP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    T     = ["T",     "T",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    # Spot single stocks matching Hyperliquid equity perps (8)
    COIN = ["COIN", "COIN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CRCL = ["CRCL", "CRCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MSTR = ["MSTR", "MSTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    PLTR = ["PLTR", "PLTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    SNDK = ["SNDK", "SNDK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    INTC = ["INTC", "INTC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    HOOD = ["HOOD", "HOOD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    SPCX = ["SPCX", "SPCX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]

    # Russell 1000 stocks bulk-added by runnable/generate_equity_universe.py (additive: appends new
    # tickers, never touches existing). Per-pair fees/base literals here are hand-editable.
    A      = ["A",      "A",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    AA     = ["AA",     "AA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    AAL    = ["AAL",    "AAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    AAON   = ["AAON",   "AAON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    ABBV   = ["ABBV",   "ABBV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    ABNB   = ["ABNB",   "ABNB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    ABT    = ["ABT",    "ABT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    ACGL   = ["ACGL",   "ACGL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    ACHC   = ["ACHC",   "ACHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    ACI    = ["ACI",    "ACI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    ACM    = ["ACM",    "ACM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ACN    = ["ACN",    "ACN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ADBE   = ["ADBE",   "ADBE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ADC    = ["ADC",    "ADC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    ADI    = ["ADI",    "ADI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ADM    = ["ADM",    "ADM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    ADP    = ["ADP",    "ADP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ADSK   = ["ADSK",   "ADSK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ADT    = ["ADT",    "ADT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    AEE    = ["AEE",    "AEE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    AEP    = ["AEP",    "AEP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    AES    = ["AES",    "AES",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    AFG    = ["AFG",    "AFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    AFL    = ["AFL",    "AFL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AFRM   = ["AFRM",   "AFRM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AGCO   = ["AGCO",   "AGCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    AGNC   = ["AGNC",   "AGNC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AGO    = ["AGO",    "AGO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    AIG    = ["AIG",    "AIG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AIT    = ["AIT",    "AIT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    AIZ    = ["AIZ",    "AIZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    AJG    = ["AJG",    "AJG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AKAM   = ["AKAM",   "AKAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ALAB   = ["ALAB",   "ALAB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ALB    = ["ALB",    "ALB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    ALGM   = ["ALGM",   "ALGM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    ALGN   = ["ALGN",   "ALGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    ALK    = ["ALK",    "ALK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ALL    = ["ALL",    "ALL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    ALLE   = ["ALLE",   "ALLE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ALLY   = ["ALLY",   "ALLY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    ALNY   = ["ALNY",   "ALNY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    ALSN   = ["ALSN",   "ALSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    AM     = ["AM",     "AM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, False]
    AMAT   = ["AMAT",   "AMAT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AMCR   = ["AMCR",   "AMCR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    AME    = ["AME",    "AME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    AMG    = ["AMG",    "AMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AMGN   = ["AMGN",   "AMGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    AMH    = ["AMH",    "AMH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    AMKR   = ["AMKR",   "AMKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AMP    = ["AMP",    "AMP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AMT    = ["AMT",    "AMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    AMTM   = ["AMTM",   "AMTM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    AN     = ["AN",     "AN",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    ANET   = ["ANET",   "ANET",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AON    = ["AON",    "AON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AOS    = ["AOS",    "AOS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    APA    = ["APA",    "APA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    APD    = ["APD",    "APD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    APG    = ["APG",    "APG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    APH    = ["APH",    "APH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    APO    = ["APO",    "APO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    APPF   = ["APPF",   "APPF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    APTV   = ["APTV",   "APTV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    AR     = ["AR",     "AR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    ARE    = ["ARE",    "ARE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    ARES   = ["ARES",   "ARES",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    ARMK   = ["ARMK",   "ARMK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    ARW    = ["ARW",    "ARW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AS     = ["AS",     "AS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    ASH    = ["ASH",    "ASH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    ASTS   = ["ASTS",   "ASTS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    ATI    = ["ATI",    "ATI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ATO    = ["ATO",    "ATO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    ATR    = ["ATR",    "ATR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    AU     = ["AU",     "AU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    AUR    = ["AUR",    "AUR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AVB    = ["AVB",    "AVB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    AVT    = ["AVT",    "AVT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    AVTR   = ["AVTR",   "AVTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    AVY    = ["AVY",    "AVY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    AWI    = ["AWI",    "AWI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    AWK    = ["AWK",    "AWK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    AXON   = ["AXON",   "AXON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    AXP    = ["AXP",    "AXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    AXS    = ["AXS",    "AXS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    AXTA   = ["AXTA",   "AXTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    AYI    = ["AYI",    "AYI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    AZO    = ["AZO",    "AZO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    BA     = ["BA",     "BA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    BAH    = ["BAH",    "BAH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    BALL   = ["BALL",   "BALL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    BAM    = ["BAM",    "BAM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    BAX    = ["BAX",    "BAX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    BBWI   = ["BBWI",   "BBWI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    BBY    = ["BBY",    "BBY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    BC     = ["BC",     "BC",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    BDX    = ["BDX",    "BDX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    BEN    = ["BEN",    "BEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    BEPC   = ["BEPC",   "BEPC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    BFAM   = ["BFAM",   "BFAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    BF_A   = ["BF_A",   "BF.A",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    BF_B   = ["BF_B",   "BF.B",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    BG     = ["BG",     "BG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    BHF    = ["BHF",    "BHF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    BIIB   = ["BIIB",   "BIIB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    BILL   = ["BILL",   "BILL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    BIO    = ["BIO",    "BIO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    BIRK   = ["BIRK",   "BIRK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    BJ     = ["BJ",     "BJ",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    BKNG   = ["BKNG",   "BKNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    BKR    = ["BKR",    "BKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    BLDR   = ["BLDR",   "BLDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    BLK    = ["BLK",    "BLK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    BLSH   = ["BLSH",   "BLSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    BMRN   = ["BMRN",   "BMRN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    BMY    = ["BMY",    "BMY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    BNY    = ["BNY",    "BNY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    BOKF   = ["BOKF",   "BOKF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    BPOP   = ["BPOP",   "BPOP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    BR     = ["BR",     "BR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    BRBR   = ["BRBR",   "BRBR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    BRKR   = ["BRKR",   "BRKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    BRO    = ["BRO",    "BRO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    BROS   = ["BROS",   "BROS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    BRX    = ["BRX",    "BRX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    BSX    = ["BSX",    "BSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    BSY    = ["BSY",    "BSY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    BURL   = ["BURL",   "BURL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    BWA    = ["BWA",    "BWA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    BWXT   = ["BWXT",   "BWXT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    BX     = ["BX",     "BX",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    BXP    = ["BXP",    "BXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    BYD    = ["BYD",    "BYD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    C      = ["C",      "C",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CACC   = ["CACC",   "CACC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    CACI   = ["CACI",   "CACI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CAG    = ["CAG",    "CAG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    CAH    = ["CAH",    "CAH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    CAI    = ["CAI",    "CAI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    CAR    = ["CAR",    "CAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CARR   = ["CARR",   "CARR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CART   = ["CART",   "CART",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    CASY   = ["CASY",   "CASY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    CAT    = ["CAT",    "CAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CAVA   = ["CAVA",   "CAVA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    CB     = ["CB",     "CB",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CBC    = ["CBC",    "CBC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    CBOE   = ["CBOE",   "CBOE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CBRE   = ["CBRE",   "CBRE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    CBSH   = ["CBSH",   "CBSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    CCC    = ["CCC",    "CCC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    CCI    = ["CCI",    "CCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    CCK    = ["CCK",    "CCK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    CCL    = ["CCL",    "CCL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    CDNS   = ["CDNS",   "CDNS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    CDW    = ["CDW",    "CDW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    CE     = ["CE",     "CE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    CEG    = ["CEG",    "CEG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    CELH   = ["CELH",   "CELH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    CERT   = ["CERT",   "CERT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    CF     = ["CF",     "CF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    CFG    = ["CFG",    "CFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CFR    = ["CFR",    "CFR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    CG     = ["CG",     "CG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CGNX   = ["CGNX",   "CGNX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    CHD    = ["CHD",    "CHD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    CHDN   = ["CHDN",   "CHDN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    CHE    = ["CHE",    "CHE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    CHH    = ["CHH",    "CHH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    CHRD   = ["CHRD",   "CHRD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    CHRW   = ["CHRW",   "CHRW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CHTR   = ["CHTR",   "CHTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    CHWY   = ["CHWY",   "CHWY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    CI     = ["CI",     "CI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    CIEN   = ["CIEN",   "CIEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    CINF   = ["CINF",   "CINF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CL     = ["CL",     "CL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    CLF    = ["CLF",    "CLF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    CLH    = ["CLH",    "CLH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CLVT   = ["CLVT",   "CLVT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    CLX    = ["CLX",    "CLX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    CMCSA  = ["CMCSA",  "CMCSA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    CME    = ["CME",    "CME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CMG    = ["CMG",    "CMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    CMI    = ["CMI",    "CMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CMS    = ["CMS",    "CMS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    CNA    = ["CNA",    "CNA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    CNC    = ["CNC",    "CNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    CNH    = ["CNH",    "CNH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CNM    = ["CNM",    "CNM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CNP    = ["CNP",    "CNP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    CNXC   = ["CNXC",   "CNXC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    COF    = ["COF",    "COF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    COHR   = ["COHR",   "COHR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    COKE   = ["COKE",   "COKE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    COLB   = ["COLB",   "COLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    COLD   = ["COLD",   "COLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    COLM   = ["COLM",   "COLM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    COO    = ["COO",    "COO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    COP    = ["COP",    "COP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    COR    = ["COR",    "COR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    CORT   = ["CORT",   "CORT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    COST   = ["COST",   "COST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    COTY   = ["COTY",   "COTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    CPAY   = ["CPAY",   "CPAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CPB    = ["CPB",    "CPB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    CPNG   = ["CPNG",   "CPNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    CPRT   = ["CPRT",   "CPRT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CPT    = ["CPT",    "CPT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    CR     = ["CR",     "CR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    CRBG   = ["CRBG",   "CRBG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CRH    = ["CRH",    "CRH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    CRL    = ["CRL",    "CRL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    CROX   = ["CROX",   "CROX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    CRS    = ["CRS",    "CRS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CRUS   = ["CRUS",   "CRUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    CRWD   = ["CRWD",   "CRWD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    CSCO   = ["CSCO",   "CSCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    CSGP   = ["CSGP",   "CSGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    CSL    = ["CSL",    "CSL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CSX    = ["CSX",    "CSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CTAS   = ["CTAS",   "CTAS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CTSH   = ["CTSH",   "CTSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    CTVA   = ["CTVA",   "CTVA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    CUBE   = ["CUBE",   "CUBE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    CUZ    = ["CUZ",    "CUZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    CVNA   = ["CVNA",   "CVNA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    CVS    = ["CVS",    "CVS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    CVX    = ["CVX",    "CVX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    CW     = ["CW",     "CW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    CWEN   = ["CWEN",   "CWEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    CXT    = ["CXT",    "CXT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    CZR    = ["CZR",    "CZR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    D      = ["D",      "D",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    DAL    = ["DAL",    "DAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    DAR    = ["DAR",    "DAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    DASH   = ["DASH",   "DASH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    DBX    = ["DBX",    "DBX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    DCI    = ["DCI",    "DCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    DD     = ["DD",     "DD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    DDOG   = ["DDOG",   "DDOG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    DDS    = ["DDS",    "DDS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    DE     = ["DE",     "DE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    DECK   = ["DECK",   "DECK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    DELL   = ["DELL",   "DELL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    DG     = ["DG",     "DG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    DGX    = ["DGX",    "DGX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    DHI    = ["DHI",    "DHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    DHR    = ["DHR",    "DHR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    DINO   = ["DINO",   "DINO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    DIS    = ["DIS",    "DIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    DJT    = ["DJT",    "DJT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    DKNG   = ["DKNG",   "DKNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    DKS    = ["DKS",    "DKS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    DLB    = ["DLB",    "DLB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    DLR    = ["DLR",    "DLR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    DLTR   = ["DLTR",   "DLTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    DOC    = ["DOC",    "DOC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    DOCS   = ["DOCS",   "DOCS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    DOCU   = ["DOCU",   "DOCU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    DOV    = ["DOV",    "DOV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    DOW    = ["DOW",    "DOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    DOX    = ["DOX",    "DOX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    DPZ    = ["DPZ",    "DPZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    DRI    = ["DRI",    "DRI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    DRS    = ["DRS",    "DRS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    DT     = ["DT",     "DT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    DTE    = ["DTE",    "DTE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    DTM    = ["DTM",    "DTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    DUK    = ["DUK",    "DUK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    DUOL   = ["DUOL",   "DUOL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    DV     = ["DV",     "DV",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    DVA    = ["DVA",    "DVA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    DVN    = ["DVN",    "DVN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    DXC    = ["DXC",    "DXC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    DXCM   = ["DXCM",   "DXCM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    EA     = ["EA",     "EA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    EBAY   = ["EBAY",   "EBAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    ECG    = ["ECG",    "ECG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    ECL    = ["ECL",    "ECL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    ED     = ["ED",     "ED",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    EEFT   = ["EEFT",   "EEFT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    EFX    = ["EFX",    "EFX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    EG     = ["EG",     "EG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    EGP    = ["EGP",    "EGP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    EHC    = ["EHC",    "EHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    EIX    = ["EIX",    "EIX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    EL     = ["EL",     "EL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    ELAN   = ["ELAN",   "ELAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    ELF    = ["ELF",    "ELF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    ELS    = ["ELS",    "ELS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    ELV    = ["ELV",    "ELV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    EME    = ["EME",    "EME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    EMN    = ["EMN",    "EMN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    EMR    = ["EMR",    "EMR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ENPH   = ["ENPH",   "ENPH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ENTG   = ["ENTG",   "ENTG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    EOG    = ["EOG",    "EOG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    EPAM   = ["EPAM",   "EPAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    EPR    = ["EPR",    "EPR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    EQH    = ["EQH",    "EQH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    EQIX   = ["EQIX",   "EQIX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    EQR    = ["EQR",    "EQR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    EQT    = ["EQT",    "EQT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    ES     = ["ES",     "ES",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    ESAB   = ["ESAB",   "ESAB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    ESI    = ["ESI",    "ESI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    ESS    = ["ESS",    "ESS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    ESTC   = ["ESTC",   "ESTC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ETN    = ["ETN",    "ETN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ETR    = ["ETR",    "ETR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    ETSY   = ["ETSY",   "ETSY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    EVR    = ["EVR",    "EVR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    EVRG   = ["EVRG",   "EVRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    EW     = ["EW",     "EW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    EWBC   = ["EWBC",   "EWBC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    EXC    = ["EXC",    "EXC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    EXE    = ["EXE",    "EXE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    EXEL   = ["EXEL",   "EXEL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    EXLS   = ["EXLS",   "EXLS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    EXP    = ["EXP",    "EXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    EXPD   = ["EXPD",   "EXPD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    EXPE   = ["EXPE",   "EXPE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    EXR    = ["EXR",    "EXR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    F      = ["F",      "F",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    FAF    = ["FAF",    "FAF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    FANG   = ["FANG",   "FANG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    FAST   = ["FAST",   "FAST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FBIN   = ["FBIN",   "FBIN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FCN    = ["FCN",    "FCN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    FCNCA  = ["FCNCA",  "FCNCA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    FCX    = ["FCX",    "FCX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    FDS    = ["FDS",    "FDS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    FDX    = ["FDX",    "FDX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FDXF   = ["FDXF",   "FDXF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FE     = ["FE",     "FE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    FERG   = ["FERG",   "FERG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FFIV   = ["FFIV",   "FFIV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    FHB    = ["FHB",    "FHB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    FHN    = ["FHN",    "FHN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    FICO   = ["FICO",   "FICO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    FIGR   = ["FIGR",   "FIGR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    FIS    = ["FIS",    "FIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    FISV   = ["FISV",   "FISV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    FITB   = ["FITB",   "FITB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    FIVE   = ["FIVE",   "FIVE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    FIX    = ["FIX",    "FIX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FLEX   = ["FLEX",   "FLEX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    FLO    = ["FLO",    "FLO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    FLS    = ["FLS",    "FLS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FLUT   = ["FLUT",   "FLUT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    FMC    = ["FMC",    "FMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    FNB    = ["FNB",    "FNB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    FND    = ["FND",    "FND",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    FNF    = ["FNF",    "FNF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    FOUR   = ["FOUR",   "FOUR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    FOX    = ["FOX",    "FOX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    FOXA   = ["FOXA",   "FOXA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    FR     = ["FR",     "FR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    FRHC   = ["FRHC",   "FRHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    FRMI   = ["FRMI",   "FRMI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    FRPT   = ["FRPT",   "FRPT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    FRT    = ["FRT",    "FRT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    FSLR   = ["FSLR",   "FSLR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    FTAI   = ["FTAI",   "FTAI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FTI    = ["FTI",    "FTI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    FTNT   = ["FTNT",   "FTNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    FTV    = ["FTV",    "FTV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    FWONA  = ["FWONA",  "FWONA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    FWONK  = ["FWONK",  "FWONK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    G      = ["G",      "G",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    GAP    = ["GAP",    "GAP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    GD     = ["GD",     "GD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    GDDY   = ["GDDY",   "GDDY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    GE     = ["GE",     "GE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    GEHC   = ["GEHC",   "GEHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    GEN    = ["GEN",    "GEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    GEV    = ["GEV",    "GEV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    GFS    = ["GFS",    "GFS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    GGG    = ["GGG",    "GGG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    GILD   = ["GILD",   "GILD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    GIS    = ["GIS",    "GIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    GL     = ["GL",     "GL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    GLIBA  = ["GLIBA",  "GLIBA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    GLIBK  = ["GLIBK",  "GLIBK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    GLOB   = ["GLOB",   "GLOB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    GLPI   = ["GLPI",   "GLPI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    GLW    = ["GLW",    "GLW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    GM     = ["GM",     "GM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    GME    = ["GME",    "GME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    GMED   = ["GMED",   "GMED",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    GNRC   = ["GNRC",   "GNRC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    GNTX   = ["GNTX",   "GNTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    GOOG   = ["GOOG",   "GOOG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    GPC    = ["GPC",    "GPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    GPK    = ["GPK",    "GPK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    GPN    = ["GPN",    "GPN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    GRMN   = ["GRMN",   "GRMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    GS     = ["GS",     "GS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    GTES   = ["GTES",   "GTES",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    GTLB   = ["GTLB",   "GTLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    GTM    = ["GTM",    "GTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    GWRE   = ["GWRE",   "GWRE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    GWW    = ["GWW",    "GWW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    GXO    = ["GXO",    "GXO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    H      = ["H",      "H",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    HAL    = ["HAL",    "HAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    HALO   = ["HALO",   "HALO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    HAS    = ["HAS",    "HAS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    HAYW   = ["HAYW",   "HAYW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    HBAN   = ["HBAN",   "HBAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    HCA    = ["HCA",    "HCA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    HEI    = ["HEI",    "HEI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    HEI_A  = ["HEI_A",  "HEI.A",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    HHH    = ["HHH",    "HHH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    HIG    = ["HIG",    "HIG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    HII    = ["HII",    "HII",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    HIW    = ["HIW",    "HIW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    HLI    = ["HLI",    "HLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    HLNE   = ["HLNE",   "HLNE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    HLT    = ["HLT",    "HLT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    HOG    = ["HOG",    "HOG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    HON    = ["HON",    "HON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    HPE    = ["HPE",    "HPE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    HPQ    = ["HPQ",    "HPQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    HR     = ["HR",     "HR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    HRB    = ["HRB",    "HRB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    HRL    = ["HRL",    "HRL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    HSIC   = ["HSIC",   "HSIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    HST    = ["HST",    "HST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    HSY    = ["HSY",    "HSY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    HUBB   = ["HUBB",   "HUBB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    HUBS   = ["HUBS",   "HUBS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    HUM    = ["HUM",    "HUM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    HUN    = ["HUN",    "HUN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    HWM    = ["HWM",    "HWM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    HXL    = ["HXL",    "HXL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    IBKR   = ["IBKR",   "IBKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    IBM    = ["IBM",    "IBM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ICE    = ["ICE",    "ICE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    IDA    = ["IDA",    "IDA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    IDXX   = ["IDXX",   "IDXX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    IEX    = ["IEX",    "IEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    IFF    = ["IFF",    "IFF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    ILMN   = ["ILMN",   "ILMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    INCY   = ["INCY",   "INCY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    INGM   = ["INGM",   "INGM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    INGR   = ["INGR",   "INGR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    INSM   = ["INSM",   "INSM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    INSP   = ["INSP",   "INSP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    INTU   = ["INTU",   "INTU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    INVH   = ["INVH",   "INVH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    IONS   = ["IONS",   "IONS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    IOT    = ["IOT",    "IOT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    IP     = ["IP",     "IP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    IPGP   = ["IPGP",   "IPGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    IQV    = ["IQV",    "IQV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    IR     = ["IR",     "IR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    IRDM   = ["IRDM",   "IRDM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    IRM    = ["IRM",    "IRM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    ISRG   = ["ISRG",   "ISRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    IT     = ["IT",     "IT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ITT    = ["ITT",    "ITT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ITW    = ["ITW",    "ITW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    IVZ    = ["IVZ",    "IVZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    J      = ["J",      "J",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    JAZZ   = ["JAZZ",   "JAZZ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    JBHT   = ["JBHT",   "JBHT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    JBL    = ["JBL",    "JBL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    JCI    = ["JCI",    "JCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    JEF    = ["JEF",    "JEF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    JHX    = ["JHX",    "JHX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    JKHY   = ["JKHY",   "JKHY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    JLL    = ["JLL",    "JLL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    JNJ    = ["JNJ",    "JNJ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    KBR    = ["KBR",    "KBR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    KD     = ["KD",     "KD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    KDP    = ["KDP",    "KDP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    KEX    = ["KEX",    "KEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    KEY    = ["KEY",    "KEY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    KEYS   = ["KEYS",   "KEYS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    KHC    = ["KHC",    "KHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    KIM    = ["KIM",    "KIM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    KKR    = ["KKR",    "KKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    KLAC   = ["KLAC",   "KLAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    KMB    = ["KMB",    "KMB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    KMI    = ["KMI",    "KMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    KMPR   = ["KMPR",   "KMPR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    KMX    = ["KMX",    "KMX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    KNSL   = ["KNSL",   "KNSL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    KNX    = ["KNX",    "KNX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    KO     = ["KO",     "KO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    KR     = ["KR",     "KR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    KRC    = ["KRC",    "KRC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    KRMN   = ["KRMN",   "KRMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    KVUE   = ["KVUE",   "KVUE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    L      = ["L",      "L",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    LAD    = ["LAD",    "LAD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    LAMR   = ["LAMR",   "LAMR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    LAZ    = ["LAZ",    "LAZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    LBRDA  = ["LBRDA",  "LBRDA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    LBRDK  = ["LBRDK",  "LBRDK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    LBTYA  = ["LBTYA",  "LBTYA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    LBTYK  = ["LBTYK",  "LBTYK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    LCID   = ["LCID",   "LCID",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    LDOS   = ["LDOS",   "LDOS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    LEA    = ["LEA",    "LEA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    LECO   = ["LECO",   "LECO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    LEN    = ["LEN",    "LEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    LEN_B  = ["LEN_B",  "LEN.B",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    LFUS   = ["LFUS",   "LFUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    LH     = ["LH",     "LH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    LHX    = ["LHX",    "LHX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    LII    = ["LII",    "LII",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    LIN    = ["LIN",    "LIN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    LINE   = ["LINE",   "LINE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    LITE   = ["LITE",   "LITE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    LKQ    = ["LKQ",    "LKQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    LLY    = ["LLY",    "LLY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    LLYVA  = ["LLYVA",  "LLYVA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    LLYVK  = ["LLYVK",  "LLYVK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    LMT    = ["LMT",    "LMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    LNC    = ["LNC",    "LNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    LNG    = ["LNG",    "LNG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    LNT    = ["LNT",    "LNT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    LOAR   = ["LOAR",   "LOAR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    LOPE   = ["LOPE",   "LOPE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    LOW    = ["LOW",    "LOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    LPLA   = ["LPLA",   "LPLA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    LPX    = ["LPX",    "LPX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    LRCX   = ["LRCX",   "LRCX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    LSCC   = ["LSCC",   "LSCC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    LSTR   = ["LSTR",   "LSTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    LULU   = ["LULU",   "LULU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    LUV    = ["LUV",    "LUV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    LVS    = ["LVS",    "LVS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    LW     = ["LW",     "LW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    LYB    = ["LYB",    "LYB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    LYFT   = ["LYFT",   "LYFT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    LYV    = ["LYV",    "LYV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    M      = ["M",      "M",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    MAA    = ["MAA",    "MAA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    MAN    = ["MAN",    "MAN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    MANH   = ["MANH",   "MANH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    MAR    = ["MAR",    "MAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    MAS    = ["MAS",    "MAS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    MAT    = ["MAT",    "MAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    MCD    = ["MCD",    "MCD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    MCHP   = ["MCHP",   "MCHP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MCK    = ["MCK",    "MCK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    MCO    = ["MCO",    "MCO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    MDB    = ["MDB",    "MDB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MDLN   = ["MDLN",   "MDLN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    MDLZ   = ["MDLZ",   "MDLZ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    MDT    = ["MDT",    "MDT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    MDU    = ["MDU",    "MDU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    MEDP   = ["MEDP",   "MEDP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    MET    = ["MET",    "MET",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    MGM    = ["MGM",    "MGM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    MHK    = ["MHK",    "MHK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    MIDD   = ["MIDD",   "MIDD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    MKC    = ["MKC",    "MKC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    MKL    = ["MKL",    "MKL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    MKSI   = ["MKSI",   "MKSI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MKTX   = ["MKTX",   "MKTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    MLI    = ["MLI",    "MLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    MLM    = ["MLM",    "MLM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    MMM    = ["MMM",    "MMM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    MNST   = ["MNST",   "MNST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    MO     = ["MO",     "MO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    MOH    = ["MOH",    "MOH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    MORN   = ["MORN",   "MORN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    MOS    = ["MOS",    "MOS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    MP     = ["MP",     "MP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    MPC    = ["MPC",    "MPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    MPT    = ["MPT",    "MPT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    MPWR   = ["MPWR",   "MPWR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MRK    = ["MRK",    "MRK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    MRNA   = ["MRNA",   "MRNA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    MRP    = ["MRP",    "MRP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    MRSH   = ["MRSH",   "MRSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    MRVL   = ["MRVL",   "MRVL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MS     = ["MS",     "MS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    MSA    = ["MSA",    "MSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    MSCI   = ["MSCI",   "MSCI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    MSGS   = ["MSGS",   "MSGS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    MSI    = ["MSI",    "MSI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MSM    = ["MSM",    "MSM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    MTB    = ["MTB",    "MTB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    MTCH   = ["MTCH",   "MTCH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    MTD    = ["MTD",    "MTD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    MTDR   = ["MTDR",   "MTDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, False]
    MTG    = ["MTG",    "MTG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    MTN    = ["MTN",    "MTN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    MTSI   = ["MTSI",   "MTSI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MTZ    = ["MTZ",    "MTZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    MUSA   = ["MUSA",   "MUSA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    NBIX   = ["NBIX",   "NBIX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    NCLH   = ["NCLH",   "NCLH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    NCNO   = ["NCNO",   "NCNO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    NDAQ   = ["NDAQ",   "NDAQ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    NDSN   = ["NDSN",   "NDSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    NEE    = ["NEE",    "NEE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    NEM    = ["NEM",    "NEM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    NET    = ["NET",    "NET",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    NEU    = ["NEU",    "NEU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    NFG    = ["NFG",    "NFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    NI     = ["NI",     "NI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    NIQ    = ["NIQ",    "NIQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    NKE    = ["NKE",    "NKE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    NLY    = ["NLY",    "NLY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    NNN    = ["NNN",    "NNN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    NOC    = ["NOC",    "NOC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    NOV    = ["NOV",    "NOV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, False]
    NOW    = ["NOW",    "NOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    NRG    = ["NRG",    "NRG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    NSA    = ["NSA",    "NSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    NSC    = ["NSC",    "NSC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    NTAP   = ["NTAP",   "NTAP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    NTNX   = ["NTNX",   "NTNX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    NTRA   = ["NTRA",   "NTRA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    NTRS   = ["NTRS",   "NTRS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    NU     = ["NU",     "NU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    NUE    = ["NUE",    "NUE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    NVR    = ["NVR",    "NVR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    NVST   = ["NVST",   "NVST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    NVT    = ["NVT",    "NVT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    NWL    = ["NWL",    "NWL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    NWS    = ["NWS",    "NWS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    NWSA   = ["NWSA",   "NWSA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    NXST   = ["NXST",   "NXST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    NYT    = ["NYT",    "NYT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    O      = ["O",      "O",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    OC     = ["OC",     "OC",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ODFL   = ["ODFL",   "ODFL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    OGE    = ["OGE",    "OGE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    OGN    = ["OGN",    "OGN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    OHI    = ["OHI",    "OHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    OKE    = ["OKE",    "OKE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    OKTA   = ["OKTA",   "OKTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    OLED   = ["OLED",   "OLED",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    OLLI   = ["OLLI",   "OLLI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    OLN    = ["OLN",    "OLN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    OMC    = ["OMC",    "OMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    OMF    = ["OMF",    "OMF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    ON     = ["ON",     "ON",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ONON   = ["ONON",   "ONON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    ONTO   = ["ONTO",   "ONTO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ORI    = ["ORI",    "ORI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    ORLY   = ["ORLY",   "ORLY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    OSK    = ["OSK",    "OSK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    OTIS   = ["OTIS",   "OTIS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    OVV    = ["OVV",    "OVV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    OWL    = ["OWL",    "OWL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    OXY    = ["OXY",    "OXY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    OZK    = ["OZK",    "OZK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    P      = ["P",      "P",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    PAG    = ["PAG",    "PAG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    PANW   = ["PANW",   "PANW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    PATH   = ["PATH",   "PATH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    PAYC   = ["PAYC",   "PAYC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    PAYX   = ["PAYX",   "PAYX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    PB     = ["PB",     "PB",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    PCAR   = ["PCAR",   "PCAR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    PCG    = ["PCG",    "PCG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    PCOR   = ["PCOR",   "PCOR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    PCTY   = ["PCTY",   "PCTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    PEG    = ["PEG",    "PEG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    PEGA   = ["PEGA",   "PEGA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    PEN    = ["PEN",    "PEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    PENN   = ["PENN",   "PENN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    PEP    = ["PEP",    "PEP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    PFE    = ["PFE",    "PFE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    PFG    = ["PFG",    "PFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    PFGC   = ["PFGC",   "PFGC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    PG     = ["PG",     "PG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    PGR    = ["PGR",    "PGR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    PH     = ["PH",     "PH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    PHM    = ["PHM",    "PHM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    PINS   = ["PINS",   "PINS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    PK     = ["PK",     "PK",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    PKG    = ["PKG",    "PKG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    PLD    = ["PLD",    "PLD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    PLNT   = ["PLNT",   "PLNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    PM     = ["PM",     "PM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    PNC    = ["PNC",    "PNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    PNFP   = ["PNFP",   "PNFP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    PNR    = ["PNR",    "PNR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    PNW    = ["PNW",    "PNW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    PODD   = ["PODD",   "PODD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    POOL   = ["POOL",   "POOL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    POST   = ["POST",   "POST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    PPC    = ["PPC",    "PPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    PPG    = ["PPG",    "PPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    PPL    = ["PPL",    "PPL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    PPLI   = ["PPLI",   "PPLI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    PR     = ["PR",     "PR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    PRGO   = ["PRGO",   "PRGO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    PRI    = ["PRI",    "PRI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    PRMB   = ["PRMB",   "PRMB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    PRU    = ["PRU",    "PRU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    PSA    = ["PSA",    "PSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    PSN    = ["PSN",    "PSN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    PSX    = ["PSX",    "PSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    PTC    = ["PTC",    "PTC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    PVH    = ["PVH",    "PVH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    PWR    = ["PWR",    "PWR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    PYPL   = ["PYPL",   "PYPL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    Q      = ["Q",      "Q",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    QCOM   = ["QCOM",   "QCOM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    QGEN   = ["QGEN",   "QGEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    QRVO   = ["QRVO",   "QRVO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    QS     = ["QS",     "QS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    QSR    = ["QSR",    "QSR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    QXO    = ["QXO",    "QXO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    R      = ["R",      "R",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    RAL    = ["RAL",    "RAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    RARE   = ["RARE",   "RARE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    RBA    = ["RBA",    "RBA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    RBC    = ["RBC",    "RBC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    RBLX   = ["RBLX",   "RBLX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    RBRK   = ["RBRK",   "RBRK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    RCL    = ["RCL",    "RCL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    RDDT   = ["RDDT",   "RDDT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    REG    = ["REG",    "REG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    REGN   = ["REGN",   "REGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    REXR   = ["REXR",   "REXR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    REYN   = ["REYN",   "REYN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    RF     = ["RF",     "RF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    RGA    = ["RGA",    "RGA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    RGEN   = ["RGEN",   "RGEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    RGLD   = ["RGLD",   "RGLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    RH     = ["RH",     "RH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    RHI    = ["RHI",    "RHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    RITM   = ["RITM",   "RITM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    RIVN   = ["RIVN",   "RIVN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    RJF    = ["RJF",    "RJF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    RKLB   = ["RKLB",   "RKLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    RKT    = ["RKT",    "RKT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    RL     = ["RL",     "RL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    RLI    = ["RLI",    "RLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    RMD    = ["RMD",    "RMD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    RNG    = ["RNG",    "RNG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    RNR    = ["RNR",    "RNR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    ROIV   = ["ROIV",   "ROIV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    ROK    = ["ROK",    "ROK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ROKU   = ["ROKU",   "ROKU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    ROL    = ["ROL",    "ROL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ROP    = ["ROP",    "ROP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ROST   = ["ROST",   "ROST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    RPM    = ["RPM",    "RPM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    RPRX   = ["RPRX",   "RPRX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    RRC    = ["RRC",    "RRC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    RRX    = ["RRX",    "RRX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    RS     = ["RS",     "RS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    RSG    = ["RSG",    "RSG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    RTX    = ["RTX",    "RTX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    RVMD   = ["RVMD",   "RVMD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    RVTY   = ["RVTY",   "RVTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    RYAN   = ["RYAN",   "RYAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    RYN    = ["RYN",    "RYN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    S      = ["S",      "S",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    SAIA   = ["SAIA",   "SAIA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    SAIC   = ["SAIC",   "SAIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    SAIL   = ["SAIL",   "SAIL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    SAM    = ["SAM",    "SAM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    SARO   = ["SARO",   "SARO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    SBAC   = ["SBAC",   "SBAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    SCCO   = ["SCCO",   "SCCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    SCHW   = ["SCHW",   "SCHW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    SCI    = ["SCI",    "SCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    SEB    = ["SEB",    "SEB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    SEIC   = ["SEIC",   "SEIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    SF     = ["SF",     "SF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    SFD    = ["SFD",    "SFD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    SFM    = ["SFM",    "SFM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    SGI    = ["SGI",    "SGI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    SHC    = ["SHC",    "SHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    SHW    = ["SHW",    "SHW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    SIRI   = ["SIRI",   "SIRI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    SITE   = ["SITE",   "SITE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    SJM    = ["SJM",    "SJM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    SLB    = ["SLB",    "SLB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    SLGN   = ["SLGN",   "SLGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    SLM    = ["SLM",    "SLM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    SMCI   = ["SMCI",   "SMCI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    SMG    = ["SMG",    "SMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    SMMT   = ["SMMT",   "SMMT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    SN     = ["SN",     "SN",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    SNA    = ["SNA",    "SNA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    SNDR   = ["SNDR",   "SNDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    SNOW   = ["SNOW",   "SNOW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    SNPS   = ["SNPS",   "SNPS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    SNX    = ["SNX",    "SNX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    SO     = ["SO",     "SO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    SOFI   = ["SOFI",   "SOFI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    SOLS   = ["SOLS",   "SOLS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    SOLV   = ["SOLV",   "SOLV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    SON    = ["SON",    "SON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    SPG    = ["SPG",    "SPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    SPGI   = ["SPGI",   "SPGI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    SPOT   = ["SPOT",   "SPOT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    SRE    = ["SRE",    "SRE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    SRPT   = ["SRPT",   "SRPT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    SSB    = ["SSB",    "SSB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    SSD    = ["SSD",    "SSD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    SSNC   = ["SSNC",   "SSNC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    ST     = ["ST",     "ST",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    STAG   = ["STAG",   "STAG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    STE    = ["STE",    "STE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    STLD   = ["STLD",   "STLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    STT    = ["STT",    "STT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    STWD   = ["STWD",   "STWD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    STZ    = ["STZ",    "STZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    SUI    = ["SUI",    "SUI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    SW     = ["SW",     "SW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    SWK    = ["SWK",    "SWK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    SWKS   = ["SWKS",   "SWKS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    SYF    = ["SYF",    "SYF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    SYK    = ["SYK",    "SYK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    SYY    = ["SYY",    "SYY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    TAP    = ["TAP",    "TAP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    TDC    = ["TDC",    "TDC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    TDG    = ["TDG",    "TDG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    TDY    = ["TDY",    "TDY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TEAM   = ["TEAM",   "TEAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TECH   = ["TECH",   "TECH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    TEM    = ["TEM",    "TEM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    TER    = ["TER",    "TER",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TFC    = ["TFC",    "TFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    TFSL   = ["TFSL",   "TFSL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    TFX    = ["TFX",    "TFX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    TGT    = ["TGT",    "TGT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    THC    = ["THC",    "THC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    THG    = ["THG",    "THG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    THO    = ["THO",    "THO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    TIGO   = ["TIGO",   "TIGO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    TJX    = ["TJX",    "TJX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    TKO    = ["TKO",    "TKO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    TKR    = ["TKR",    "TKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    TLN    = ["TLN",    "TLN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    TMO    = ["TMO",    "TMO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    TMUS   = ["TMUS",   "TMUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    TNL    = ["TNL",    "TNL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    TOL    = ["TOL",    "TOL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    TOST   = ["TOST",   "TOST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    TPG    = ["TPG",    "TPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    TPL    = ["TPL",    "TPL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    TPR    = ["TPR",    "TPR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    TREX   = ["TREX",   "TREX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    TRGP   = ["TRGP",   "TRGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    TRMB   = ["TRMB",   "TRMB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TROW   = ["TROW",   "TROW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    TRU    = ["TRU",    "TRU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    TRV    = ["TRV",    "TRV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    TSCO   = ["TSCO",   "TSCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    TSN    = ["TSN",    "TSN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    TT     = ["TT",     "TT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    TTC    = ["TTC",    "TTC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    TTD    = ["TTD",    "TTD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    TTEK   = ["TTEK",   "TTEK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    TTWO   = ["TTWO",   "TTWO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    TW     = ["TW",     "TW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    TWLO   = ["TWLO",   "TWLO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TXN    = ["TXN",    "TXN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TXRH   = ["TXRH",   "TXRH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    TXT    = ["TXT",    "TXT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    TYL    = ["TYL",    "TYL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    U      = ["U",      "U",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    UA     = ["UA",     "UA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    UAA    = ["UAA",    "UAA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    UAL    = ["UAL",    "UAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    UDR    = ["UDR",    "UDR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    UGI    = ["UGI",    "UGI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    UHAL   = ["UHAL",   "UHAL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    UHAL_B = ["UHAL_B", "UHAL.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    UHS    = ["UHS",    "UHS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    UI     = ["UI",     "UI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    ULTA   = ["ULTA",   "ULTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    UNH    = ["UNH",    "UNH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    UNM    = ["UNM",    "UNM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    UNP    = ["UNP",    "UNP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    UPS    = ["UPS",    "UPS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    URI    = ["URI",    "URI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    USB    = ["USB",    "USB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    USFD   = ["USFD",   "USFD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    UTHR   = ["UTHR",   "UTHR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    UWMC   = ["UWMC",   "UWMC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    VEEV   = ["VEEV",   "VEEV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    VFC    = ["VFC",    "VFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    VGNT   = ["VGNT",   "VGNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    VICI   = ["VICI",   "VICI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    VIK    = ["VIK",    "VIK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    VIRT   = ["VIRT",   "VIRT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    VKTX   = ["VKTX",   "VKTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    VLO    = ["VLO",    "VLO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    VLTO   = ["VLTO",   "VLTO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    VMC    = ["VMC",    "VMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    VMI    = ["VMI",    "VMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    VNO    = ["VNO",    "VNO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    VNOM   = ["VNOM",   "VNOM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, False]
    VNT    = ["VNT",    "VNT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    VOYA   = ["VOYA",   "VOYA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    VRSK   = ["VRSK",   "VRSK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    VRSN   = ["VRSN",   "VRSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    VRT    = ["VRT",    "VRT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    VRTX   = ["VRTX",   "VRTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    VSNT   = ["VSNT",   "VSNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    VST    = ["VST",    "VST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    VTR    = ["VTR",    "VTR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    VTRS   = ["VTRS",   "VTRS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    VVV    = ["VVV",    "VVV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    VZ     = ["VZ",     "VZ",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    W      = ["W",      "W",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    WAB    = ["WAB",    "WAB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    WAL    = ["WAL",    "WAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    WAT    = ["WAT",    "WAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    WBD    = ["WBD",    "WBD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    WBS    = ["WBS",    "WBS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    WCC    = ["WCC",    "WCC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    WDAY   = ["WDAY",   "WDAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    WDC    = ["WDC",    "WDC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    WEC    = ["WEC",    "WEC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    WELL   = ["WELL",   "WELL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    WEN    = ["WEN",    "WEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    WEX    = ["WEX",    "WEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    WFC    = ["WFC",    "WFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    WFRD   = ["WFRD",   "WFRD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    WH     = ["WH",     "WH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    WHR    = ["WHR",    "WHR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    WING   = ["WING",   "WING",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    WLK    = ["WLK",    "WLK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    WM     = ["WM",     "WM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    WMB    = ["WMB",    "WMB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    WMS    = ["WMS",    "WMS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    WMT    = ["WMT",    "WMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    WPC    = ["WPC",    "WPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    WRB    = ["WRB",    "WRB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    WSC    = ["WSC",    "WSC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    WSM    = ["WSM",    "WSM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    WSO    = ["WSO",    "WSO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    WST    = ["WST",    "WST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    WTFC   = ["WTFC",   "WTFC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    WTM    = ["WTM",    "WTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    WTRG   = ["WTRG",   "WTRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    WTW    = ["WTW",    "WTW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    WU     = ["WU",     "WU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    WWD    = ["WWD",    "WWD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    WY     = ["WY",     "WY",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    WYNN   = ["WYNN",   "WYNN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    XEL    = ["XEL",    "XEL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    XOM    = ["XOM",    "XOM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    XP     = ["XP",     "XP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    XPO    = ["XPO",    "XPO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    XRAY   = ["XRAY",   "XRAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    XYL    = ["XYL",    "XYL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    XYZ    = ["XYZ",    "XYZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    YETI   = ["YETI",   "YETI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    YUM    = ["YUM",    "YUM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    Z      = ["Z",      "Z",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    ZBH    = ["ZBH",    "ZBH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    ZBRA   = ["ZBRA",   "ZBRA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ZG     = ["ZG",     "ZG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]
    ZION   = ["ZION",   "ZION",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    ZM     = ["ZM",     "ZM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ZS     = ["ZS",     "ZS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    ZTS    = ["ZTS",    "ZTS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]

    # Equities - Sector ETFs (22)
    XLK  = ["XLK",  "XLK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    VGT  = ["VGT",  "VGT",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, False]
    XLF  = ["XLF",  "XLF",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    VFH  = ["VFH",  "VFH",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, False]
    XLY  = ["XLY",  "XLY",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    VCR  = ["VCR",  "VCR",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, False]
    XLC  = ["XLC",  "XLC",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    VOX  = ["VOX",  "VOX",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, False]
    XLV  = ["XLV",  "XLV",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, True]
    VHT  = ["VHT",  "VHT",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE, False]
    XLI  = ["XLI",  "XLI",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, True]
    VIS  = ["VIS",  "VIS",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS, False]
    XLP  = ["XLP",  "XLP",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, True]
    VDC  = ["VDC",  "VDC",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES, False]
    XLE  = ["XLE",  "XLE",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, True]
    VDE  = ["VDE",  "VDE",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY, False]
    XLB  = ["XLB",  "XLB",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, True]
    VAW  = ["VAW",  "VAW",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS, False]
    XLU  = ["XLU",  "XLU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, True]
    VPU  = ["VPU",  "VPU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES, False]
    XLRE = ["XLRE", "XLRE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, True]
    VNQ  = ["VNQ",  "VNQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE, False]

    # Index ETFs (broad market & international)
    SPY  = ["SPY",  "SPY",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]
    QQQ  = ["QQQ",  "QQQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]
    DIA  = ["DIA",  "DIA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]
    IWM  = ["IWM",  "IWM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]
    EWU  = ["EWU",  "EWU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    EWG  = ["EWG",  "EWG",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    EWJ  = ["EWJ",  "EWJ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]
    EWH  = ["EWH",  "EWH",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    EWA  = ["EWA",  "EWA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    EWQ  = ["EWQ",  "EWQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), False]
    EFA  = ["EFA",  "EFA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]
    IEMG = ["IEMG", "IEMG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]
    INDA = ["INDA", "INDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]
    VT   = ["VT",   "VT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), True]

    # indices (no longer allowed for trading as we moved to equities tickers instead)
    SPX   = ["SPX",   "SPX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]
    DJI   = ["DJI",   "DJI",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]
    NDX   = ["NDX",   "NDX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]
    VIX   = ["VIX",   "VIX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]
    FTSE  = ["FTSE",  "FTSE",  0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]
    GDAXI = ["GDAXI", "GDAXI", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5), False]

    # Hyperliquid Trade Pairs (USDC-quoted, src=HYPERLIQUID)
    # Crypto perp futures
    BTCUSDC   = ["BTCUSDC",   "BTC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    ETHUSDC   = ["ETHUSDC",   "ETH/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    SOLUSDC   = ["SOLUSDC",   "SOL/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    BNBUSDC   = ["BNBUSDC",   "BNB/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    XRPUSDC   = ["XRPUSDC",   "XRP/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    DOGEUSDC  = ["DOGEUSDC",  "DOGE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    ADAUSDC   = ["ADAUSDC",   "ADA/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    AVAXUSDC  = ["AVAXUSDC",  "AVAX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    LINKUSDC  = ["LINKUSDC",  "LINK/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    DOTUSDC   = ["DOTUSDC",   "DOT/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    TONUSDC   = ["TONUSDC",   "TON/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    TRXUSDC   = ["TRXUSDC",   "TRX/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    LTCUSDC   = ["LTCUSDC",   "LTC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    BCHUSDC   = ["BCHUSDC",   "BCH/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    TAOUSDC   = ["TAOUSDC",   "TAO/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    SUIUSDC   = ["SUIUSDC",   "SUI/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    ARBUSDC   = ["ARBUSDC",   "ARB/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    NEARUSDC  = ["NEARUSDC",  "NEAR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    ALGOUSDC  = ["ALGOUSDC",  "ALGO/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    ASTERUSDC = ["ASTERUSDC", "ASTER/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    UNIUSDC   = ["UNIUSDC",   "UNI/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    AAVEUSDC  = ["AAVEUSDC",  "AAVE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    CRVUSDC   = ["CRVUSDC",   "CRV/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    HYPEUSDC  = ["HYPEUSDC",  "HYPE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    XMRUSDC   = ["XMRUSDC",   "XMR/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    ZECUSDC   = ["ZECUSDC",   "ZEC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    PAXGUSDC  = ["PAXGUSDC",  "PAXG/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    ENAUSDC   = ["ENAUSDC",   "ENA/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    ZROUSDC   = ["ZROUSDC",   "ZRO/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    WLDUSDC   = ["WLDUSDC",   "WLD/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    PUMPUSDC  = ["PUMPUSDC",  "PUMP/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    KPEPEUSDC = ["kPEPEUSDC", "kPEPE/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]

    # Commodity perp futures (synthetic, track commodity prices — not physical delivery)
    WTIOILUSDC   = ["WTIOILUSDC",   "WTIOIL/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:CL",       InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    BRENTOILUSDC = ["BRENTOILUSDC", "BRENTOIL/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:BRENTOIL", InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), False]
    GOLDUSDC     = ["GOLDUSDC",     "GOLD/USDC",     0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:GOLD",     InstrumentType.PERP, SubaccountTierBaseLeverage(1.0), True]
    SILVERUSDC   = ["SILVERUSDC",   "SILVER/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:SILVER",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    COPPERUSDC   = ["COPPERUSDC",   "COPPER/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:COPPER",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    NATGASUSDC   = ["NATGASUSDC",   "NATGAS/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:NATGAS",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]
    PLATINUMUSDC = ["PLATINUMUSDC", "PLATINUM/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:PLATINUM", InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), True]

    # Index perp futures (synthetic, track equity index prices — not ETFs)
    SP500USDC  = ["SP500USDC",  "SP500/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:SP500",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.5), True]
    XYZ100USDC = ["XYZ100USDC", "XYZ100/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:XYZ100", InstrumentType.PERP, SubaccountTierBaseLeverage(1.5), True]
    EWYUSDC    = ["EWYUSDC",    "EWY/USDC",    0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:EWY",    InstrumentType.PERP, SubaccountTierBaseLeverage(1.5), True]

    # Equity perp futures (synthetic, track single-stock prices — not actual shares)
    NVDAUSDC  = ["NVDAUSDC",  "NVDA/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:NVDA",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AAPLUSDC  = ["AAPLUSDC",  "AAPL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AAPL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TSLAUSDC  = ["TSLAUSDC",  "TSLA/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:TSLA",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    MSFTUSDC  = ["MSFTUSDC",  "MSFT/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MSFT",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AMZNUSDC  = ["AMZNUSDC",  "AMZN/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AMZN",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY, True]
    GOOGLUSDC = ["GOOGLUSDC", "GOOGL/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:GOOGL", InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    METAUSDC  = ["METAUSDC",  "META/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:META",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    COINUSDC  = ["COINUSDC",  "COIN/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:COIN",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    CRCLUSDC  = ["CRCLUSDC",  "CRCL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:CRCL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MSTRUSDC  = ["MSTRUSDC",  "MSTR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MSTR",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    PLTRUSDC  = ["PLTRUSDC",  "PLTR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:PLTR",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    AMDUSDC   = ["AMDUSDC",   "AMD/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AMD",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    TSMUSDC   = ["TSMUSDC",   "TSM/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:TSM",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    NFLXUSDC  = ["NFLXUSDC",  "NFLX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:NFLX",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]
    SNDKUSDC  = ["SNDKUSDC",  "SNDK/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:SNDK",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    INTCUSDC  = ["INTCUSDC",  "INTC/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:INTC",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    MUUSDC    = ["MUUSDC",    "MU/USDC",    0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MU",    InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    HOODUSDC  = ["HOODUSDC",  "HOOD/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:HOOD",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS, True]
    ORCLUSDC  = ["ORCLUSDC",  "ORCL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:ORCL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY, True]
    SPCXUSDC  = ["SPCXUSDC",  "SPCX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:SPCX",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION, True]

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
        # type() is str (not isinstance) to exclude InstrumentType/ExposureGroup, str subclasses via str-Enum.
        if self.src == TradePairSource.HYPERLIQUID and len(self.value) > 8 and type(self.value[8]) is str:
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
        """Per-pair Tier-1 base for the legacy subaccount tier curve.
        See leverage_utils.get_legacy_tier_positional_leverage.
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

    def carry_fee_rate_per_interval(self, is_pro=False) -> float:
        if self.src == TradePairSource.HYPERLIQUID:
            return 0
        rates = PRO_CARRY_FEE_RATE_PER_INTERVAL if is_pro else CARRY_FEE_RATE_PER_INTERVAL
        return rates.get(self.trade_pair_category, 0)

    @property
    def exposure_group(self) -> "ExposureGroup | None":
        """Correlated-exposure group, or None for pairs that belong to no group.

        Located by type scan, like instrument_type — position-independent.
        """
        for v in self.value:
            if isinstance(v, ExposureGroup):
                return v
        return None

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
    def is_pro(self) -> bool:
        """True if pro accounts may trade this pair.

        Pro accounts trade a curated subset of the standard universe: names below the
        liquidity floor, duplicate listings, and deprecated Vanta-native crypto are excluded
        (reviewed quarterly). Located by type scan, like instrument_type — the bool is the
        only one in the value list, so the lookup is position-independent.
        """
        for v in self.value:
            if isinstance(v, bool):
                return v
        raise ValueError(f"TradePair {self.trade_pair_id} is missing is_pro")

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
    TradePair.BCHUSD:  TradePair.BCHUSDC,
    TradePair.LINKUSD: TradePair.LINKUSDC,
    TradePair.XMRUSD:  TradePair.XMRUSDC,
    TradePair.LTCUSD:  TradePair.LTCUSDC,
}
