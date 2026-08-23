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

# Trade pairs a pro account may trade, on top of its asset class. None means no pro-specific
# restriction — pro accounts trade whatever their asset class permits.
PRO_ALLOWED_TRADE_PAIR_IDS = None


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
    AUDCAD = ["AUDCAD", "AUD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    AUDCHF = ["AUDCHF", "AUD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    AUDUSD = ["AUDUSD", "AUD/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    AUDJPY = ["AUDJPY", "AUD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    AUDNZD = ["AUDNZD", "AUD/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    CADCHF = ["CADCHF", "CAD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    CADJPY = ["CADJPY", "CAD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    CHFJPY = ["CHFJPY", "CHF/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    EURAUD = ["EURAUD", "EUR/AUD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    EURCAD = ["EURCAD", "EUR/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    EURUSD = ["EURUSD", "EUR/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    EURCHF = ["EURCHF", "EUR/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    EURGBP = ["EURGBP", "EUR/GBP", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    EURJPY = ["EURJPY", "EUR/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    EURNZD = ["EURNZD", "EUR/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G3, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    NZDCAD = ["NZDCAD", "NZD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    NZDCHF = ["NZDCHF", "NZD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    NZDJPY = ["NZDJPY", "NZD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    NZDUSD = ["NZDUSD", "NZD/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GBPAUD = ["GBPAUD", "GBP/AUD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GBPCAD = ["GBPCAD", "GBP/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GBPCHF = ["GBPCHF", "GBP/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GBPJPY = ["GBPJPY", "GBP/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G2, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GBPNZD = ["GBPNZD", "GBP/NZD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G4, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GBPUSD = ["GBPUSD", "GBP/USD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    USDCAD = ["USDCAD", "USD/CAD", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    USDCHF = ["USDCHF", "USD/CHF", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    USDJPY = ["USDJPY", "USD/JPY", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G1, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    USDMXN = ["USDMXN", "USD/MXN", 0.00007, FOREX_MIN_LEVERAGE, FOREX_MAX_LEVERAGE, TradePairCategory.FOREX, ForexSubcategory.G5, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]


    # "Commodities" (Bundle with Forex for now)
    XAUUSD = ["XAUUSD", "XAU/USD", 0.00007, COMMODITIES_MIN_LEVERAGE, COMMODITIES_MAX_LEVERAGE, TradePairCategory.FOREX, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    XAGUSD = ["XAGUSD", "XAG/USD", 0.00007, COMMODITIES_MIN_LEVERAGE, COMMODITIES_MAX_LEVERAGE, TradePairCategory.FOREX, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]

    # Equities - Stocks
    # Technology (10)
    NVDA = ["NVDA", "NVDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MSFT = ["MSFT", "MSFT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AAPL = ["AAPL", "AAPL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AVGO = ["AVGO", "AVGO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TSM  = ["TSM",  "TSM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ORCL = ["ORCL", "ORCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AMD  = ["AMD",  "AMD",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MU   = ["MU",   "MU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CRM  = ["CRM",  "CRM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    UBER = ["UBER", "UBER", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    # Financial Services (5)
    BRK_B = ["BRK_B", "BRK.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    JPM   = ["JPM",   "JPM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    V     = ["V",     "V",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MA    = ["MA",    "MA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BAC   = ["BAC",   "BAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    # Consumer Discretionary (5)
    AMZN = ["AMZN", "AMZN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    TSLA = ["TSLA", "TSLA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    HD   = ["HD",   "HD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BABA = ["BABA", "BABA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    SBUX = ["SBUX", "SBUX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    # Communication Services (5)
    GOOGL = ["GOOGL", "GOOGL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    META  = ["META",  "META",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    NFLX  = ["NFLX",  "NFLX",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    APP   = ["APP",   "APP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    T     = ["T",     "T",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    # Spot single stocks matching Hyperliquid equity perps (8)
    COIN = ["COIN", "COIN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CRCL = ["CRCL", "CRCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MSTR = ["MSTR", "MSTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    PLTR = ["PLTR", "PLTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SNDK = ["SNDK", "SNDK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    INTC = ["INTC", "INTC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    HOOD = ["HOOD", "HOOD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SPCX = ["SPCX", "SPCX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]

    # Russell 1000 stocks bulk-added by runnable/generate_equity_universe.py (additive: appends new
    # tickers, never touches existing). Per-pair fees/base literals here are hand-editable.
    A      = ["A",      "A",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    AA     = ["AA",     "AA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    AAL    = ["AAL",    "AAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AAON   = ["AAON",   "AAON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ABBV   = ["ABBV",   "ABBV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    ABNB   = ["ABNB",   "ABNB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    ABT    = ["ABT",    "ABT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    ACGL   = ["ACGL",   "ACGL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ACHC   = ["ACHC",   "ACHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    ACI    = ["ACI",    "ACI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    ACM    = ["ACM",    "ACM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ACN    = ["ACN",    "ACN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ADBE   = ["ADBE",   "ADBE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ADC    = ["ADC",    "ADC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    ADI    = ["ADI",    "ADI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ADM    = ["ADM",    "ADM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    ADP    = ["ADP",    "ADP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ADSK   = ["ADSK",   "ADSK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ADT    = ["ADT",    "ADT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    AEE    = ["AEE",    "AEE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    AEP    = ["AEP",    "AEP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    AES    = ["AES",    "AES",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    AFG    = ["AFG",    "AFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AFL    = ["AFL",    "AFL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AFRM   = ["AFRM",   "AFRM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AGCO   = ["AGCO",   "AGCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AGNC   = ["AGNC",   "AGNC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AGO    = ["AGO",    "AGO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AIG    = ["AIG",    "AIG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AIT    = ["AIT",    "AIT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AIZ    = ["AIZ",    "AIZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AJG    = ["AJG",    "AJG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AKAM   = ["AKAM",   "AKAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ALAB   = ["ALAB",   "ALAB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ALB    = ["ALB",    "ALB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    ALGM   = ["ALGM",   "ALGM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ALGN   = ["ALGN",   "ALGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    ALK    = ["ALK",    "ALK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ALL    = ["ALL",    "ALL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ALLE   = ["ALLE",   "ALLE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ALLY   = ["ALLY",   "ALLY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ALNY   = ["ALNY",   "ALNY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    ALSN   = ["ALSN",   "ALSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AM     = ["AM",     "AM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    AMAT   = ["AMAT",   "AMAT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AMCR   = ["AMCR",   "AMCR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    AME    = ["AME",    "AME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AMG    = ["AMG",    "AMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AMGN   = ["AMGN",   "AMGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    AMH    = ["AMH",    "AMH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    AMKR   = ["AMKR",   "AMKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AMP    = ["AMP",    "AMP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AMT    = ["AMT",    "AMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    AMTM   = ["AMTM",   "AMTM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AN     = ["AN",     "AN",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    ANET   = ["ANET",   "ANET",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AON    = ["AON",    "AON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AOS    = ["AOS",    "AOS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    APA    = ["APA",    "APA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    APD    = ["APD",    "APD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    APG    = ["APG",    "APG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    APH    = ["APH",    "APH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    APO    = ["APO",    "APO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    APPF   = ["APPF",   "APPF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    APTV   = ["APTV",   "APTV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    AR     = ["AR",     "AR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    ARE    = ["ARE",    "ARE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    ARES   = ["ARES",   "ARES",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ARMK   = ["ARMK",   "ARMK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    ARW    = ["ARW",    "ARW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AS     = ["AS",     "AS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    ASH    = ["ASH",    "ASH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    ASTS   = ["ASTS",   "ASTS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    ATI    = ["ATI",    "ATI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ATO    = ["ATO",    "ATO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    ATR    = ["ATR",    "ATR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    AU     = ["AU",     "AU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    AUR    = ["AUR",    "AUR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AVB    = ["AVB",    "AVB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    AVT    = ["AVT",    "AVT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AVTR   = ["AVTR",   "AVTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    AVY    = ["AVY",    "AVY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    AWI    = ["AWI",    "AWI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AWK    = ["AWK",    "AWK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    AXON   = ["AXON",   "AXON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AXP    = ["AXP",    "AXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AXS    = ["AXS",    "AXS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    AXTA   = ["AXTA",   "AXTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    AYI    = ["AYI",    "AYI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    AZO    = ["AZO",    "AZO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BA     = ["BA",     "BA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    BAH    = ["BAH",    "BAH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    BALL   = ["BALL",   "BALL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    BAM    = ["BAM",    "BAM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BAX    = ["BAX",    "BAX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    BBWI   = ["BBWI",   "BBWI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BBY    = ["BBY",    "BBY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BC     = ["BC",     "BC",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BDX    = ["BDX",    "BDX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    BEN    = ["BEN",    "BEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BEPC   = ["BEPC",   "BEPC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    BFAM   = ["BFAM",   "BFAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BF_A   = ["BF_A",   "BF.A",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    BF_B   = ["BF_B",   "BF.B",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    BG     = ["BG",     "BG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    BHF    = ["BHF",    "BHF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BIIB   = ["BIIB",   "BIIB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    BILL   = ["BILL",   "BILL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    BIO    = ["BIO",    "BIO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    BIRK   = ["BIRK",   "BIRK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BJ     = ["BJ",     "BJ",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    BKNG   = ["BKNG",   "BKNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BKR    = ["BKR",    "BKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    BLDR   = ["BLDR",   "BLDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    BLK    = ["BLK",    "BLK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BLSH   = ["BLSH",   "BLSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BMRN   = ["BMRN",   "BMRN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    BMY    = ["BMY",    "BMY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    BNY    = ["BNY",    "BNY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BOKF   = ["BOKF",   "BOKF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BPOP   = ["BPOP",   "BPOP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BR     = ["BR",     "BR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    BRBR   = ["BRBR",   "BRBR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    BRKR   = ["BRKR",   "BRKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    BRO    = ["BRO",    "BRO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BROS   = ["BROS",   "BROS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BRX    = ["BRX",    "BRX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    BSX    = ["BSX",    "BSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    BSY    = ["BSY",    "BSY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    BURL   = ["BURL",   "BURL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BWA    = ["BWA",    "BWA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    BWXT   = ["BWXT",   "BWXT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    BX     = ["BX",     "BX",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    BXP    = ["BXP",    "BXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    BYD    = ["BYD",    "BYD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    C      = ["C",      "C",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CACC   = ["CACC",   "CACC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CACI   = ["CACI",   "CACI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CAG    = ["CAG",    "CAG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CAH    = ["CAH",    "CAH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CAI    = ["CAI",    "CAI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CAR    = ["CAR",    "CAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CARR   = ["CARR",   "CARR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CART   = ["CART",   "CART",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CASY   = ["CASY",   "CASY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CAT    = ["CAT",    "CAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CAVA   = ["CAVA",   "CAVA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CB     = ["CB",     "CB",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CBC    = ["CBC",    "CBC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CBOE   = ["CBOE",   "CBOE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CBRE   = ["CBRE",   "CBRE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    CBSH   = ["CBSH",   "CBSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CCC    = ["CCC",    "CCC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CCI    = ["CCI",    "CCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    CCK    = ["CCK",    "CCK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    CCL    = ["CCL",    "CCL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CDNS   = ["CDNS",   "CDNS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CDW    = ["CDW",    "CDW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CE     = ["CE",     "CE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    CEG    = ["CEG",    "CEG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    CELH   = ["CELH",   "CELH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CERT   = ["CERT",   "CERT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CF     = ["CF",     "CF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    CFG    = ["CFG",    "CFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CFR    = ["CFR",    "CFR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CG     = ["CG",     "CG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CGNX   = ["CGNX",   "CGNX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CHD    = ["CHD",    "CHD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CHDN   = ["CHDN",   "CHDN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CHE    = ["CHE",    "CHE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CHH    = ["CHH",    "CHH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CHRD   = ["CHRD",   "CHRD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    CHRW   = ["CHRW",   "CHRW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CHTR   = ["CHTR",   "CHTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    CHWY   = ["CHWY",   "CHWY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CI     = ["CI",     "CI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CIEN   = ["CIEN",   "CIEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CINF   = ["CINF",   "CINF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CL     = ["CL",     "CL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CLF    = ["CLF",    "CLF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    CLH    = ["CLH",    "CLH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CLVT   = ["CLVT",   "CLVT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CLX    = ["CLX",    "CLX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CMCSA  = ["CMCSA",  "CMCSA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    CME    = ["CME",    "CME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CMG    = ["CMG",    "CMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CMI    = ["CMI",    "CMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CMS    = ["CMS",    "CMS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    CNA    = ["CNA",    "CNA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CNC    = ["CNC",    "CNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CNH    = ["CNH",    "CNH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CNM    = ["CNM",    "CNM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CNP    = ["CNP",    "CNP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    CNXC   = ["CNXC",   "CNXC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    COF    = ["COF",    "COF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    COHR   = ["COHR",   "COHR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    COKE   = ["COKE",   "COKE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    COLB   = ["COLB",   "COLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    COLD   = ["COLD",   "COLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    COLM   = ["COLM",   "COLM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    COO    = ["COO",    "COO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    COP    = ["COP",    "COP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    COR    = ["COR",    "COR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CORT   = ["CORT",   "CORT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    COST   = ["COST",   "COST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    COTY   = ["COTY",   "COTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CPAY   = ["CPAY",   "CPAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CPB    = ["CPB",    "CPB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    CPNG   = ["CPNG",   "CPNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CPRT   = ["CPRT",   "CPRT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CPT    = ["CPT",    "CPT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    CR     = ["CR",     "CR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CRBG   = ["CRBG",   "CRBG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CRH    = ["CRH",    "CRH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    CRL    = ["CRL",    "CRL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CROX   = ["CROX",   "CROX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CRS    = ["CRS",    "CRS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CRUS   = ["CRUS",   "CRUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CRWD   = ["CRWD",   "CRWD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CSCO   = ["CSCO",   "CSCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CSGP   = ["CSGP",   "CSGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    CSL    = ["CSL",    "CSL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CSX    = ["CSX",    "CSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CTAS   = ["CTAS",   "CTAS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CTSH   = ["CTSH",   "CTSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CTVA   = ["CTVA",   "CTVA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    CUBE   = ["CUBE",   "CUBE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    CUZ    = ["CUZ",    "CUZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    CVNA   = ["CVNA",   "CVNA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    CVS    = ["CVS",    "CVS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    CVX    = ["CVX",    "CVX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    CW     = ["CW",     "CW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    CWEN   = ["CWEN",   "CWEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    CXT    = ["CXT",    "CXT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    CZR    = ["CZR",    "CZR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    D      = ["D",      "D",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    DAL    = ["DAL",    "DAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    DAR    = ["DAR",    "DAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    DASH   = ["DASH",   "DASH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DBX    = ["DBX",    "DBX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    DCI    = ["DCI",    "DCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    DD     = ["DD",     "DD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    DDOG   = ["DDOG",   "DDOG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    DDS    = ["DDS",    "DDS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DE     = ["DE",     "DE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    DECK   = ["DECK",   "DECK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DELL   = ["DELL",   "DELL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    DG     = ["DG",     "DG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    DGX    = ["DGX",    "DGX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    DHI    = ["DHI",    "DHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DHR    = ["DHR",    "DHR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    DINO   = ["DINO",   "DINO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    DIS    = ["DIS",    "DIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    DJT    = ["DJT",    "DJT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    DKNG   = ["DKNG",   "DKNG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DKS    = ["DKS",    "DKS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DLB    = ["DLB",    "DLB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    DLR    = ["DLR",    "DLR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    DLTR   = ["DLTR",   "DLTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    DOC    = ["DOC",    "DOC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    DOCS   = ["DOCS",   "DOCS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    DOCU   = ["DOCU",   "DOCU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    DOV    = ["DOV",    "DOV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    DOW    = ["DOW",    "DOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    DOX    = ["DOX",    "DOX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    DPZ    = ["DPZ",    "DPZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DRI    = ["DRI",    "DRI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DRS    = ["DRS",    "DRS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    DT     = ["DT",     "DT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    DTE    = ["DTE",    "DTE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    DTM    = ["DTM",    "DTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    DUK    = ["DUK",    "DUK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    DUOL   = ["DUOL",   "DUOL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    DV     = ["DV",     "DV",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    DVA    = ["DVA",    "DVA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    DVN    = ["DVN",    "DVN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    DXC    = ["DXC",    "DXC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    DXCM   = ["DXCM",   "DXCM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    EA     = ["EA",     "EA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    EBAY   = ["EBAY",   "EBAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    ECG    = ["ECG",    "ECG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ECL    = ["ECL",    "ECL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    ED     = ["ED",     "ED",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    EEFT   = ["EEFT",   "EEFT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    EFX    = ["EFX",    "EFX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    EG     = ["EG",     "EG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    EGP    = ["EGP",    "EGP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    EHC    = ["EHC",    "EHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    EIX    = ["EIX",    "EIX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    EL     = ["EL",     "EL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    ELAN   = ["ELAN",   "ELAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    ELF    = ["ELF",    "ELF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    ELS    = ["ELS",    "ELS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    ELV    = ["ELV",    "ELV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    EME    = ["EME",    "EME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    EMN    = ["EMN",    "EMN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    EMR    = ["EMR",    "EMR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ENPH   = ["ENPH",   "ENPH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ENTG   = ["ENTG",   "ENTG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    EOG    = ["EOG",    "EOG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    EPAM   = ["EPAM",   "EPAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    EPR    = ["EPR",    "EPR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    EQH    = ["EQH",    "EQH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    EQIX   = ["EQIX",   "EQIX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    EQR    = ["EQR",    "EQR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    EQT    = ["EQT",    "EQT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    ES     = ["ES",     "ES",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    ESAB   = ["ESAB",   "ESAB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ESI    = ["ESI",    "ESI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    ESS    = ["ESS",    "ESS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    ESTC   = ["ESTC",   "ESTC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ETN    = ["ETN",    "ETN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ETR    = ["ETR",    "ETR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    ETSY   = ["ETSY",   "ETSY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    EVR    = ["EVR",    "EVR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    EVRG   = ["EVRG",   "EVRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    EW     = ["EW",     "EW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    EWBC   = ["EWBC",   "EWBC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    EXC    = ["EXC",    "EXC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    EXE    = ["EXE",    "EXE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    EXEL   = ["EXEL",   "EXEL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    EXLS   = ["EXLS",   "EXLS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    EXP    = ["EXP",    "EXP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    EXPD   = ["EXPD",   "EXPD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    EXPE   = ["EXPE",   "EXPE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    EXR    = ["EXR",    "EXR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    F      = ["F",      "F",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    FAF    = ["FAF",    "FAF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FANG   = ["FANG",   "FANG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    FAST   = ["FAST",   "FAST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FBIN   = ["FBIN",   "FBIN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FCN    = ["FCN",    "FCN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FCNCA  = ["FCNCA",  "FCNCA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FCX    = ["FCX",    "FCX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    FDS    = ["FDS",    "FDS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FDX    = ["FDX",    "FDX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FDXF   = ["FDXF",   "FDXF",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FE     = ["FE",     "FE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    FERG   = ["FERG",   "FERG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FFIV   = ["FFIV",   "FFIV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    FHB    = ["FHB",    "FHB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FHN    = ["FHN",    "FHN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FICO   = ["FICO",   "FICO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    FIGR   = ["FIGR",   "FIGR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FIS    = ["FIS",    "FIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FISV   = ["FISV",   "FISV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FITB   = ["FITB",   "FITB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FIVE   = ["FIVE",   "FIVE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    FIX    = ["FIX",    "FIX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FLEX   = ["FLEX",   "FLEX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    FLO    = ["FLO",    "FLO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    FLS    = ["FLS",    "FLS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FLUT   = ["FLUT",   "FLUT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    FMC    = ["FMC",    "FMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    FNB    = ["FNB",    "FNB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FND    = ["FND",    "FND",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    FNF    = ["FNF",    "FNF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FOUR   = ["FOUR",   "FOUR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FOX    = ["FOX",    "FOX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    FOXA   = ["FOXA",   "FOXA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    FR     = ["FR",     "FR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    FRHC   = ["FRHC",   "FRHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    FRMI   = ["FRMI",   "FRMI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    FRPT   = ["FRPT",   "FRPT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    FRT    = ["FRT",    "FRT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    FSLR   = ["FSLR",   "FSLR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    FTAI   = ["FTAI",   "FTAI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FTI    = ["FTI",    "FTI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    FTNT   = ["FTNT",   "FTNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    FTV    = ["FTV",    "FTV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    FWONA  = ["FWONA",  "FWONA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    FWONK  = ["FWONK",  "FWONK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    G      = ["G",      "G",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    GAP    = ["GAP",    "GAP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    GD     = ["GD",     "GD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    GDDY   = ["GDDY",   "GDDY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    GE     = ["GE",     "GE",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    GEHC   = ["GEHC",   "GEHC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    GEN    = ["GEN",    "GEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    GEV    = ["GEV",    "GEV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    GFS    = ["GFS",    "GFS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    GGG    = ["GGG",    "GGG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    GILD   = ["GILD",   "GILD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    GIS    = ["GIS",    "GIS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    GL     = ["GL",     "GL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    GLIBA  = ["GLIBA",  "GLIBA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    GLIBK  = ["GLIBK",  "GLIBK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    GLOB   = ["GLOB",   "GLOB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    GLPI   = ["GLPI",   "GLPI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    GLW    = ["GLW",    "GLW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    GM     = ["GM",     "GM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    GME    = ["GME",    "GME",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    GMED   = ["GMED",   "GMED",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    GNRC   = ["GNRC",   "GNRC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    GNTX   = ["GNTX",   "GNTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    GOOG   = ["GOOG",   "GOOG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    GPC    = ["GPC",    "GPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    GPK    = ["GPK",    "GPK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    GPN    = ["GPN",    "GPN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    GRMN   = ["GRMN",   "GRMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    GS     = ["GS",     "GS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    GTES   = ["GTES",   "GTES",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    GTLB   = ["GTLB",   "GTLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    GTM    = ["GTM",    "GTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    GWRE   = ["GWRE",   "GWRE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    GWW    = ["GWW",    "GWW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    GXO    = ["GXO",    "GXO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    H      = ["H",      "H",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    HAL    = ["HAL",    "HAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    HALO   = ["HALO",   "HALO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    HAS    = ["HAS",    "HAS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    HAYW   = ["HAYW",   "HAYW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    HBAN   = ["HBAN",   "HBAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    HCA    = ["HCA",    "HCA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    HEI    = ["HEI",    "HEI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    HEI_A  = ["HEI_A",  "HEI.A",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    HHH    = ["HHH",    "HHH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    HIG    = ["HIG",    "HIG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    HII    = ["HII",    "HII",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    HIW    = ["HIW",    "HIW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    HLI    = ["HLI",    "HLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    HLNE   = ["HLNE",   "HLNE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    HLT    = ["HLT",    "HLT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    HOG    = ["HOG",    "HOG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    HON    = ["HON",    "HON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    HPE    = ["HPE",    "HPE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    HPQ    = ["HPQ",    "HPQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    HR     = ["HR",     "HR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    HRB    = ["HRB",    "HRB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    HRL    = ["HRL",    "HRL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    HSIC   = ["HSIC",   "HSIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    HST    = ["HST",    "HST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    HSY    = ["HSY",    "HSY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    HUBB   = ["HUBB",   "HUBB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    HUBS   = ["HUBS",   "HUBS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    HUM    = ["HUM",    "HUM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    HUN    = ["HUN",    "HUN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    HWM    = ["HWM",    "HWM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    HXL    = ["HXL",    "HXL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    IBKR   = ["IBKR",   "IBKR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    IBM    = ["IBM",    "IBM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ICE    = ["ICE",    "ICE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    IDA    = ["IDA",    "IDA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    IDXX   = ["IDXX",   "IDXX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    IEX    = ["IEX",    "IEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    IFF    = ["IFF",    "IFF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    ILMN   = ["ILMN",   "ILMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    INCY   = ["INCY",   "INCY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    INGM   = ["INGM",   "INGM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    INGR   = ["INGR",   "INGR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    INSM   = ["INSM",   "INSM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    INSP   = ["INSP",   "INSP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    INTU   = ["INTU",   "INTU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    INVH   = ["INVH",   "INVH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    IONS   = ["IONS",   "IONS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    IOT    = ["IOT",    "IOT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    IP     = ["IP",     "IP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    IPGP   = ["IPGP",   "IPGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    IQV    = ["IQV",    "IQV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    IR     = ["IR",     "IR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    IRDM   = ["IRDM",   "IRDM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    IRM    = ["IRM",    "IRM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    ISRG   = ["ISRG",   "ISRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    IT     = ["IT",     "IT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ITT    = ["ITT",    "ITT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ITW    = ["ITW",    "ITW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    IVZ    = ["IVZ",    "IVZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    J      = ["J",      "J",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    JAZZ   = ["JAZZ",   "JAZZ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    JBHT   = ["JBHT",   "JBHT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    JBL    = ["JBL",    "JBL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    JCI    = ["JCI",    "JCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    JEF    = ["JEF",    "JEF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    JHX    = ["JHX",    "JHX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    JKHY   = ["JKHY",   "JKHY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    JLL    = ["JLL",    "JLL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    JNJ    = ["JNJ",    "JNJ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    KBR    = ["KBR",    "KBR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    KD     = ["KD",     "KD",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    KDP    = ["KDP",    "KDP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    KEX    = ["KEX",    "KEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    KEY    = ["KEY",    "KEY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    KEYS   = ["KEYS",   "KEYS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    KHC    = ["KHC",    "KHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    KIM    = ["KIM",    "KIM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    KKR    = ["KKR",    "KKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    KLAC   = ["KLAC",   "KLAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    KMB    = ["KMB",    "KMB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    KMI    = ["KMI",    "KMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    KMPR   = ["KMPR",   "KMPR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    KMX    = ["KMX",    "KMX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    KNSL   = ["KNSL",   "KNSL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    KNX    = ["KNX",    "KNX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    KO     = ["KO",     "KO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    KR     = ["KR",     "KR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    KRC    = ["KRC",    "KRC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    KRMN   = ["KRMN",   "KRMN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    KVUE   = ["KVUE",   "KVUE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    L      = ["L",      "L",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    LAD    = ["LAD",    "LAD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LAMR   = ["LAMR",   "LAMR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    LAZ    = ["LAZ",    "LAZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    LBRDA  = ["LBRDA",  "LBRDA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    LBRDK  = ["LBRDK",  "LBRDK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    LBTYA  = ["LBTYA",  "LBTYA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    LBTYK  = ["LBTYK",  "LBTYK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    LCID   = ["LCID",   "LCID",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LDOS   = ["LDOS",   "LDOS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LEA    = ["LEA",    "LEA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LECO   = ["LECO",   "LECO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LEN    = ["LEN",    "LEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LEN_B  = ["LEN_B",  "LEN.B",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LFUS   = ["LFUS",   "LFUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    LH     = ["LH",     "LH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    LHX    = ["LHX",    "LHX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LII    = ["LII",    "LII",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LIN    = ["LIN",    "LIN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    LINE   = ["LINE",   "LINE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    LITE   = ["LITE",   "LITE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    LKQ    = ["LKQ",    "LKQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LLY    = ["LLY",    "LLY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    LLYVA  = ["LLYVA",  "LLYVA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LLYVK  = ["LLYVK",  "LLYVK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LMT    = ["LMT",    "LMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LNC    = ["LNC",    "LNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    LNG    = ["LNG",    "LNG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    LNT    = ["LNT",    "LNT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    LOAR   = ["LOAR",   "LOAR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LOPE   = ["LOPE",   "LOPE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LOW    = ["LOW",    "LOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LPLA   = ["LPLA",   "LPLA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    LPX    = ["LPX",    "LPX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    LRCX   = ["LRCX",   "LRCX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    LSCC   = ["LSCC",   "LSCC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    LSTR   = ["LSTR",   "LSTR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LULU   = ["LULU",   "LULU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LUV    = ["LUV",    "LUV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LVS    = ["LVS",    "LVS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    LW     = ["LW",     "LW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    LYB    = ["LYB",    "LYB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    LYFT   = ["LYFT",   "LYFT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    LYV    = ["LYV",    "LYV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    M      = ["M",      "M",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    MAA    = ["MAA",    "MAA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    MAN    = ["MAN",    "MAN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    MANH   = ["MANH",   "MANH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MAR    = ["MAR",    "MAR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    MAS    = ["MAS",    "MAS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    MAT    = ["MAT",    "MAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    MCD    = ["MCD",    "MCD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    MCHP   = ["MCHP",   "MCHP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MCK    = ["MCK",    "MCK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    MCO    = ["MCO",    "MCO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MDB    = ["MDB",    "MDB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MDLN   = ["MDLN",   "MDLN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    MDLZ   = ["MDLZ",   "MDLZ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    MDT    = ["MDT",    "MDT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    MDU    = ["MDU",    "MDU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    MEDP   = ["MEDP",   "MEDP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    MET    = ["MET",    "MET",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MGM    = ["MGM",    "MGM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    MHK    = ["MHK",    "MHK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    MIDD   = ["MIDD",   "MIDD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    MKC    = ["MKC",    "MKC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    MKL    = ["MKL",    "MKL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MKSI   = ["MKSI",   "MKSI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MKTX   = ["MKTX",   "MKTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MLI    = ["MLI",    "MLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    MLM    = ["MLM",    "MLM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    MMM    = ["MMM",    "MMM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    MNST   = ["MNST",   "MNST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    MO     = ["MO",     "MO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    MOH    = ["MOH",    "MOH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    MORN   = ["MORN",   "MORN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MOS    = ["MOS",    "MOS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    MP     = ["MP",     "MP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    MPC    = ["MPC",    "MPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    MPT    = ["MPT",    "MPT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    MPWR   = ["MPWR",   "MPWR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MRK    = ["MRK",    "MRK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    MRNA   = ["MRNA",   "MRNA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    MRP    = ["MRP",    "MRP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    MRSH   = ["MRSH",   "MRSH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MRVL   = ["MRVL",   "MRVL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MS     = ["MS",     "MS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MSA    = ["MSA",    "MSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    MSCI   = ["MSCI",   "MSCI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MSGS   = ["MSGS",   "MSGS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    MSI    = ["MSI",    "MSI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MSM    = ["MSM",    "MSM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    MTB    = ["MTB",    "MTB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MTCH   = ["MTCH",   "MTCH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    MTD    = ["MTD",    "MTD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    MTDR   = ["MTDR",   "MTDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    MTG    = ["MTG",    "MTG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    MTN    = ["MTN",    "MTN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    MTSI   = ["MTSI",   "MTSI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MTZ    = ["MTZ",    "MTZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    MUSA   = ["MUSA",   "MUSA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    NBIX   = ["NBIX",   "NBIX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    NCLH   = ["NCLH",   "NCLH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    NCNO   = ["NCNO",   "NCNO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    NDAQ   = ["NDAQ",   "NDAQ",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    NDSN   = ["NDSN",   "NDSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    NEE    = ["NEE",    "NEE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    NEM    = ["NEM",    "NEM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    NET    = ["NET",    "NET",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    NEU    = ["NEU",    "NEU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    NFG    = ["NFG",    "NFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    NI     = ["NI",     "NI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    NIQ    = ["NIQ",    "NIQ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    NKE    = ["NKE",    "NKE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    NLY    = ["NLY",    "NLY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    NNN    = ["NNN",    "NNN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    NOC    = ["NOC",    "NOC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    NOV    = ["NOV",    "NOV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    NOW    = ["NOW",    "NOW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    NRG    = ["NRG",    "NRG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    NSA    = ["NSA",    "NSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    NSC    = ["NSC",    "NSC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    NTAP   = ["NTAP",   "NTAP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    NTNX   = ["NTNX",   "NTNX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    NTRA   = ["NTRA",   "NTRA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    NTRS   = ["NTRS",   "NTRS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    NU     = ["NU",     "NU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    NUE    = ["NUE",    "NUE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    NVR    = ["NVR",    "NVR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    NVST   = ["NVST",   "NVST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    NVT    = ["NVT",    "NVT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    NWL    = ["NWL",    "NWL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    NWS    = ["NWS",    "NWS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    NWSA   = ["NWSA",   "NWSA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    NXST   = ["NXST",   "NXST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    NYT    = ["NYT",    "NYT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    O      = ["O",      "O",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    OC     = ["OC",     "OC",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ODFL   = ["ODFL",   "ODFL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    OGE    = ["OGE",    "OGE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    OGN    = ["OGN",    "OGN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    OHI    = ["OHI",    "OHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    OKE    = ["OKE",    "OKE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    OKTA   = ["OKTA",   "OKTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    OLED   = ["OLED",   "OLED",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    OLLI   = ["OLLI",   "OLLI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    OLN    = ["OLN",    "OLN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    OMC    = ["OMC",    "OMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    OMF    = ["OMF",    "OMF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ON     = ["ON",     "ON",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ONON   = ["ONON",   "ONON",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    ONTO   = ["ONTO",   "ONTO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ORI    = ["ORI",    "ORI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ORLY   = ["ORLY",   "ORLY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    OSK    = ["OSK",    "OSK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    OTIS   = ["OTIS",   "OTIS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    OVV    = ["OVV",    "OVV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    OWL    = ["OWL",    "OWL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    OXY    = ["OXY",    "OXY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    OZK    = ["OZK",    "OZK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    P      = ["P",      "P",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    PAG    = ["PAG",    "PAG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    PANW   = ["PANW",   "PANW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    PATH   = ["PATH",   "PATH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    PAYC   = ["PAYC",   "PAYC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    PAYX   = ["PAYX",   "PAYX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    PB     = ["PB",     "PB",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    PCAR   = ["PCAR",   "PCAR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    PCG    = ["PCG",    "PCG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    PCOR   = ["PCOR",   "PCOR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    PCTY   = ["PCTY",   "PCTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    PEG    = ["PEG",    "PEG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    PEGA   = ["PEGA",   "PEGA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    PEN    = ["PEN",    "PEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    PENN   = ["PENN",   "PENN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    PEP    = ["PEP",    "PEP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    PFE    = ["PFE",    "PFE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    PFG    = ["PFG",    "PFG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    PFGC   = ["PFGC",   "PFGC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    PG     = ["PG",     "PG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    PGR    = ["PGR",    "PGR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    PH     = ["PH",     "PH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    PHM    = ["PHM",    "PHM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    PINS   = ["PINS",   "PINS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    PK     = ["PK",     "PK",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    PKG    = ["PKG",    "PKG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    PLD    = ["PLD",    "PLD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    PLNT   = ["PLNT",   "PLNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    PM     = ["PM",     "PM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    PNC    = ["PNC",    "PNC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    PNFP   = ["PNFP",   "PNFP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    PNR    = ["PNR",    "PNR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    PNW    = ["PNW",    "PNW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    PODD   = ["PODD",   "PODD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    POOL   = ["POOL",   "POOL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    POST   = ["POST",   "POST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    PPC    = ["PPC",    "PPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    PPG    = ["PPG",    "PPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    PPL    = ["PPL",    "PPL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    PPLI   = ["PPLI",   "PPLI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    PR     = ["PR",     "PR",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    PRGO   = ["PRGO",   "PRGO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    PRI    = ["PRI",    "PRI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    PRMB   = ["PRMB",   "PRMB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    PRU    = ["PRU",    "PRU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    PSA    = ["PSA",    "PSA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    PSN    = ["PSN",    "PSN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    PSX    = ["PSX",    "PSX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    PTC    = ["PTC",    "PTC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    PVH    = ["PVH",    "PVH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    PWR    = ["PWR",    "PWR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    PYPL   = ["PYPL",   "PYPL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    Q      = ["Q",      "Q",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    QCOM   = ["QCOM",   "QCOM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    QGEN   = ["QGEN",   "QGEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    QRVO   = ["QRVO",   "QRVO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    QS     = ["QS",     "QS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    QSR    = ["QSR",    "QSR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    QXO    = ["QXO",    "QXO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    R      = ["R",      "R",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    RAL    = ["RAL",    "RAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    RARE   = ["RARE",   "RARE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    RBA    = ["RBA",    "RBA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    RBC    = ["RBC",    "RBC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    RBLX   = ["RBLX",   "RBLX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    RBRK   = ["RBRK",   "RBRK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    RCL    = ["RCL",    "RCL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    RDDT   = ["RDDT",   "RDDT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    REG    = ["REG",    "REG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    REGN   = ["REGN",   "REGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    REXR   = ["REXR",   "REXR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    REYN   = ["REYN",   "REYN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    RF     = ["RF",     "RF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    RGA    = ["RGA",    "RGA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    RGEN   = ["RGEN",   "RGEN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    RGLD   = ["RGLD",   "RGLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    RH     = ["RH",     "RH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    RHI    = ["RHI",    "RHI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    RITM   = ["RITM",   "RITM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    RIVN   = ["RIVN",   "RIVN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    RJF    = ["RJF",    "RJF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    RKLB   = ["RKLB",   "RKLB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    RKT    = ["RKT",    "RKT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    RL     = ["RL",     "RL",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    RLI    = ["RLI",    "RLI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    RMD    = ["RMD",    "RMD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    RNG    = ["RNG",    "RNG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    RNR    = ["RNR",    "RNR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ROIV   = ["ROIV",   "ROIV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    ROK    = ["ROK",    "ROK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ROKU   = ["ROKU",   "ROKU",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    ROL    = ["ROL",    "ROL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ROP    = ["ROP",    "ROP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ROST   = ["ROST",   "ROST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    RPM    = ["RPM",    "RPM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    RPRX   = ["RPRX",   "RPRX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    RRC    = ["RRC",    "RRC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    RRX    = ["RRX",    "RRX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    RS     = ["RS",     "RS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    RSG    = ["RSG",    "RSG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    RTX    = ["RTX",    "RTX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    RVMD   = ["RVMD",   "RVMD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    RVTY   = ["RVTY",   "RVTY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    RYAN   = ["RYAN",   "RYAN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    RYN    = ["RYN",    "RYN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    S      = ["S",      "S",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SAIA   = ["SAIA",   "SAIA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    SAIC   = ["SAIC",   "SAIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    SAIL   = ["SAIL",   "SAIL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SAM    = ["SAM",    "SAM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    SARO   = ["SARO",   "SARO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    SBAC   = ["SBAC",   "SBAC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    SCCO   = ["SCCO",   "SCCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    SCHW   = ["SCHW",   "SCHW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SCI    = ["SCI",    "SCI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    SEB    = ["SEB",    "SEB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    SEIC   = ["SEIC",   "SEIC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SF     = ["SF",     "SF",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SFD    = ["SFD",    "SFD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    SFM    = ["SFM",    "SFM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    SGI    = ["SGI",    "SGI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    SHC    = ["SHC",    "SHC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    SHW    = ["SHW",    "SHW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    SIRI   = ["SIRI",   "SIRI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    SITE   = ["SITE",   "SITE",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    SJM    = ["SJM",    "SJM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    SLB    = ["SLB",    "SLB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    SLGN   = ["SLGN",   "SLGN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    SLM    = ["SLM",    "SLM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SMCI   = ["SMCI",   "SMCI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SMG    = ["SMG",    "SMG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    SMMT   = ["SMMT",   "SMMT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    SN     = ["SN",     "SN",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    SNA    = ["SNA",    "SNA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    SNDR   = ["SNDR",   "SNDR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    SNOW   = ["SNOW",   "SNOW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SNPS   = ["SNPS",   "SNPS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SNX    = ["SNX",    "SNX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SO     = ["SO",     "SO",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    SOFI   = ["SOFI",   "SOFI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SOLS   = ["SOLS",   "SOLS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    SOLV   = ["SOLV",   "SOLV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    SON    = ["SON",    "SON",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    SPG    = ["SPG",    "SPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    SPGI   = ["SPGI",   "SPGI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SPOT   = ["SPOT",   "SPOT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    SRE    = ["SRE",    "SRE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    SRPT   = ["SRPT",   "SRPT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    SSB    = ["SSB",    "SSB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SSD    = ["SSD",    "SSD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    SSNC   = ["SSNC",   "SSNC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    ST     = ["ST",     "ST",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    STAG   = ["STAG",   "STAG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    STE    = ["STE",    "STE",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    STLD   = ["STLD",   "STLD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    STT    = ["STT",    "STT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    STWD   = ["STWD",   "STWD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    STZ    = ["STZ",    "STZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    SUI    = ["SUI",    "SUI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    SW     = ["SW",     "SW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    SWK    = ["SWK",    "SWK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    SWKS   = ["SWKS",   "SWKS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SYF    = ["SYF",    "SYF",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    SYK    = ["SYK",    "SYK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    SYY    = ["SYY",    "SYY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    TAP    = ["TAP",    "TAP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    TDC    = ["TDC",    "TDC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TDG    = ["TDG",    "TDG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    TDY    = ["TDY",    "TDY",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TEAM   = ["TEAM",   "TEAM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TECH   = ["TECH",   "TECH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    TEM    = ["TEM",    "TEM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    TER    = ["TER",    "TER",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TFC    = ["TFC",    "TFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    TFSL   = ["TFSL",   "TFSL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    TFX    = ["TFX",    "TFX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    TGT    = ["TGT",    "TGT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    THC    = ["THC",    "THC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    THG    = ["THG",    "THG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    THO    = ["THO",    "THO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    TIGO   = ["TIGO",   "TIGO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    TJX    = ["TJX",    "TJX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    TKO    = ["TKO",    "TKO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    TKR    = ["TKR",    "TKR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    TLN    = ["TLN",    "TLN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    TMO    = ["TMO",    "TMO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    TMUS   = ["TMUS",   "TMUS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    TNL    = ["TNL",    "TNL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    TOL    = ["TOL",    "TOL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    TOST   = ["TOST",   "TOST",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    TPG    = ["TPG",    "TPG",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    TPL    = ["TPL",    "TPL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    TPR    = ["TPR",    "TPR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    TREX   = ["TREX",   "TREX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    TRGP   = ["TRGP",   "TRGP",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    TRMB   = ["TRMB",   "TRMB",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TROW   = ["TROW",   "TROW",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    TRU    = ["TRU",    "TRU",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    TRV    = ["TRV",    "TRV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    TSCO   = ["TSCO",   "TSCO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    TSN    = ["TSN",    "TSN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    TT     = ["TT",     "TT",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    TTC    = ["TTC",    "TTC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    TTD    = ["TTD",    "TTD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    TTEK   = ["TTEK",   "TTEK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    TTWO   = ["TTWO",   "TTWO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    TW     = ["TW",     "TW",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    TWLO   = ["TWLO",   "TWLO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TXN    = ["TXN",    "TXN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TXRH   = ["TXRH",   "TXRH",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    TXT    = ["TXT",    "TXT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    TYL    = ["TYL",    "TYL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    U      = ["U",      "U",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    UA     = ["UA",     "UA",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    UAA    = ["UAA",    "UAA",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    UAL    = ["UAL",    "UAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    UDR    = ["UDR",    "UDR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    UGI    = ["UGI",    "UGI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    UHAL   = ["UHAL",   "UHAL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    UHAL_B = ["UHAL_B", "UHAL.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    UHS    = ["UHS",    "UHS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    UI     = ["UI",     "UI",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ULTA   = ["ULTA",   "ULTA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    UNH    = ["UNH",    "UNH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    UNM    = ["UNM",    "UNM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    UNP    = ["UNP",    "UNP",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    UPS    = ["UPS",    "UPS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    URI    = ["URI",    "URI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    USB    = ["USB",    "USB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    USFD   = ["USFD",   "USFD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    UTHR   = ["UTHR",   "UTHR",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    UWMC   = ["UWMC",   "UWMC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    VEEV   = ["VEEV",   "VEEV",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    VFC    = ["VFC",    "VFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    VGNT   = ["VGNT",   "VGNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    VICI   = ["VICI",   "VICI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    VIK    = ["VIK",    "VIK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    VIRT   = ["VIRT",   "VIRT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    VKTX   = ["VKTX",   "VKTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    VLO    = ["VLO",    "VLO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    VLTO   = ["VLTO",   "VLTO",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    VMC    = ["VMC",    "VMC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    VMI    = ["VMI",    "VMI",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    VNO    = ["VNO",    "VNO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    VNOM   = ["VNOM",   "VNOM",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    VNT    = ["VNT",    "VNT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    VOYA   = ["VOYA",   "VOYA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    VRSK   = ["VRSK",   "VRSK",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    VRSN   = ["VRSN",   "VRSN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    VRT    = ["VRT",    "VRT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    VRTX   = ["VRTX",   "VRTX",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    VSNT   = ["VSNT",   "VSNT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    VST    = ["VST",    "VST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    VTR    = ["VTR",    "VTR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    VTRS   = ["VTRS",   "VTRS",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    VVV    = ["VVV",    "VVV",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    VZ     = ["VZ",     "VZ",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    W      = ["W",      "W",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    WAB    = ["WAB",    "WAB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    WAL    = ["WAL",    "WAL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WAT    = ["WAT",    "WAT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    WBD    = ["WBD",    "WBD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    WBS    = ["WBS",    "WBS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WCC    = ["WCC",    "WCC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    WDAY   = ["WDAY",   "WDAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    WDC    = ["WDC",    "WDC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    WEC    = ["WEC",    "WEC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    WELL   = ["WELL",   "WELL",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    WEN    = ["WEN",    "WEN",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    WEX    = ["WEX",    "WEX",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WFC    = ["WFC",    "WFC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WFRD   = ["WFRD",   "WFRD",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    WH     = ["WH",     "WH",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    WHR    = ["WHR",    "WHR",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    WING   = ["WING",   "WING",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    WLK    = ["WLK",    "WLK",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    WM     = ["WM",     "WM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    WMB    = ["WMB",    "WMB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    WMS    = ["WMS",    "WMS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    WMT    = ["WMT",    "WMT",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    WPC    = ["WPC",    "WPC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    WRB    = ["WRB",    "WRB",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WSC    = ["WSC",    "WSC",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    WSM    = ["WSM",    "WSM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    WSO    = ["WSO",    "WSO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    WST    = ["WST",    "WST",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    WTFC   = ["WTFC",   "WTFC",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WTM    = ["WTM",    "WTM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WTRG   = ["WTRG",   "WTRG",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    WTW    = ["WTW",    "WTW",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WU     = ["WU",     "WU",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    WWD    = ["WWD",    "WWD",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    WY     = ["WY",     "WY",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    WYNN   = ["WYNN",   "WYNN",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    XEL    = ["XEL",    "XEL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    XOM    = ["XOM",    "XOM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    XP     = ["XP",     "XP",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    XPO    = ["XPO",    "XPO",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    XRAY   = ["XRAY",   "XRAY",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    XYL    = ["XYL",    "XYL",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    XYZ    = ["XYZ",    "XYZ",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    YETI   = ["YETI",   "YETI",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    YUM    = ["YUM",    "YUM",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    Z      = ["Z",      "Z",      0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    ZBH    = ["ZBH",    "ZBH",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    ZBRA   = ["ZBRA",   "ZBRA",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ZG     = ["ZG",     "ZG",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    ZION   = ["ZION",   "ZION",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ZM     = ["ZM",     "ZM",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ZS     = ["ZS",     "ZS",     0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    ZTS    = ["ZTS",    "ZTS",    0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]

    # Equities - Sector ETFs (22)
    XLK  = ["XLK",  "XLK",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    VGT  = ["VGT",  "VGT",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    XLF  = ["XLF",  "XLF",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    VFH  = ["VFH",  "VFH",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    XLY  = ["XLY",  "XLY",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    VCR  = ["VCR",  "VCR",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    XLC  = ["XLC",  "XLC",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    VOX  = ["VOX",  "VOX",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    XLV  = ["XLV",  "XLV",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    VHT  = ["VHT",  "VHT",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.HEALTH_CARE]
    XLI  = ["XLI",  "XLI",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    VIS  = ["VIS",  "VIS",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.INDUSTRIALS]
    XLP  = ["XLP",  "XLP",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    VDC  = ["VDC",  "VDC",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_STAPLES]
    XLE  = ["XLE",  "XLE",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    VDE  = ["VDE",  "VDE",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.ENERGY]
    XLB  = ["XLB",  "XLB",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    VAW  = ["VAW",  "VAW",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.MATERIALS]
    XLU  = ["XLU",  "XLU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    VPU  = ["VPU",  "VPU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.UTILITIES]
    XLRE = ["XLRE", "XLRE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]
    VNQ  = ["VNQ",  "VNQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5), ExposureGroup.REAL_ESTATE]

    # Index ETFs (broad market & international)
    SPY  = ["SPY",  "SPY",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    QQQ  = ["QQQ",  "QQQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DIA  = ["DIA",  "DIA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IWM  = ["IWM",  "IWM",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EWU  = ["EWU",  "EWU",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EWG  = ["EWG",  "EWG",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EWJ  = ["EWJ",  "EWJ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EWH  = ["EWH",  "EWH",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EWA  = ["EWA",  "EWA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EWQ  = ["EWQ",  "EWQ",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EFA  = ["EFA",  "EFA",  0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IEMG = ["IEMG", "IEMG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INDA = ["INDA", "INDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VT   = ["VT",   "VT",   0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]

    # indices (no longer allowed for trading as we moved to equities tickers instead)
    SPX   = ["SPX",   "SPX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    DJI   = ["DJI",   "DJI",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    NDX   = ["NDX",   "NDX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    VIX   = ["VIX",   "VIX",   0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    FTSE  = ["FTSE",  "FTSE",  0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GDAXI = ["GDAXI", "GDAXI", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]

    # Hyperliquid Trade Pairs (USDC-quoted, src=HYPERLIQUID)
    # Crypto perp futures
    BTCUSDC   = ["BTCUSDC",   "BTC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ETHUSDC   = ["ETHUSDC",   "ETH/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    SOLUSDC   = ["SOLUSDC",   "SOL/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    BNBUSDC   = ["BNBUSDC",   "BNB/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    XRPUSDC   = ["XRPUSDC",   "XRP/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    DOGEUSDC  = ["DOGEUSDC",  "DOGE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ADAUSDC   = ["ADAUSDC",   "ADA/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    AVAXUSDC  = ["AVAXUSDC",  "AVAX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    LINKUSDC  = ["LINKUSDC",  "LINK/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    DOTUSDC   = ["DOTUSDC",   "DOT/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TONUSDC   = ["TONUSDC",   "TON/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TRXUSDC   = ["TRXUSDC",   "TRX/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    LTCUSDC   = ["LTCUSDC",   "LTC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    BCHUSDC   = ["BCHUSDC",   "BCH/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TAOUSDC   = ["TAOUSDC",   "TAO/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    SUIUSDC   = ["SUIUSDC",   "SUI/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ARBUSDC   = ["ARBUSDC",   "ARB/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    NEARUSDC  = ["NEARUSDC",  "NEAR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ALGOUSDC  = ["ALGOUSDC",  "ALGO/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ASTERUSDC = ["ASTERUSDC", "ASTER/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    UNIUSDC   = ["UNIUSDC",   "UNI/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    AAVEUSDC  = ["AAVEUSDC",  "AAVE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    CRVUSDC   = ["CRVUSDC",   "CRV/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    HYPEUSDC  = ["HYPEUSDC",  "HYPE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    XMRUSDC   = ["XMRUSDC",   "XMR/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ZECUSDC   = ["ZECUSDC",   "ZEC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    PAXGUSDC  = ["PAXGUSDC",  "PAXG/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ENAUSDC   = ["ENAUSDC",   "ENA/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ZROUSDC   = ["ZROUSDC",   "ZRO/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    WLDUSDC   = ["WLDUSDC",   "WLD/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    PUMPUSDC  = ["PUMPUSDC",  "PUMP/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    KPEPEUSDC = ["kPEPEUSDC", "kPEPE/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO, None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]

    # Commodity perp futures (synthetic, track commodity prices — not physical delivery)
    WTIOILUSDC   = ["WTIOILUSDC",   "WTIOIL/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:CL",       InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    BRENTOILUSDC = ["BRENTOILUSDC", "BRENTOIL/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:BRENTOIL", InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    GOLDUSDC     = ["GOLDUSDC",     "GOLD/USDC",     0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:GOLD",     InstrumentType.PERP, SubaccountTierBaseLeverage(1.0)]
    SILVERUSDC   = ["SILVERUSDC",   "SILVER/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:SILVER",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    COPPERUSDC   = ["COPPERUSDC",   "COPPER/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:COPPER",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    NATGASUSDC   = ["NATGASUSDC",   "NATGAS/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:NATGAS",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    PLATINUMUSDC = ["PLATINUMUSDC", "PLATINUM/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.COMMODITIES, None, TradePairSource.HYPERLIQUID, "xyz:PLATINUM", InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]

    # Index perp futures (synthetic, track equity index prices — not ETFs)
    SP500USDC  = ["SP500USDC",  "SP500/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:SP500",  InstrumentType.PERP, SubaccountTierBaseLeverage(1.5)]
    XYZ100USDC = ["XYZ100USDC", "XYZ100/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:XYZ100", InstrumentType.PERP, SubaccountTierBaseLeverage(1.5)]
    EWYUSDC    = ["EWYUSDC",    "EWY/USDC",    0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.INDICES, None, TradePairSource.HYPERLIQUID, "xyz:EWY",    InstrumentType.PERP, SubaccountTierBaseLeverage(1.5)]

    # Equity perp futures (synthetic, track single-stock prices — not actual shares)
    NVDAUSDC  = ["NVDAUSDC",  "NVDA/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:NVDA",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AAPLUSDC  = ["AAPLUSDC",  "AAPL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AAPL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TSLAUSDC  = ["TSLAUSDC",  "TSLA/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:TSLA",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    MSFTUSDC  = ["MSFTUSDC",  "MSFT/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MSFT",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AMZNUSDC  = ["AMZNUSDC",  "AMZN/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AMZN",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.CONSUMER_DISCRETIONARY]
    GOOGLUSDC = ["GOOGLUSDC", "GOOGL/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:GOOGL", InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    METAUSDC  = ["METAUSDC",  "META/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:META",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    COINUSDC  = ["COINUSDC",  "COIN/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:COIN",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    CRCLUSDC  = ["CRCLUSDC",  "CRCL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:CRCL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MSTRUSDC  = ["MSTRUSDC",  "MSTR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MSTR",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    PLTRUSDC  = ["PLTRUSDC",  "PLTR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:PLTR",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    AMDUSDC   = ["AMDUSDC",   "AMD/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AMD",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    TSMUSDC   = ["TSMUSDC",   "TSM/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:TSM",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    NFLXUSDC  = ["NFLXUSDC",  "NFLX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:NFLX",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]
    SNDKUSDC  = ["SNDKUSDC",  "SNDK/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:SNDK",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    INTCUSDC  = ["INTCUSDC",  "INTC/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:INTC",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    MUUSDC    = ["MUUSDC",    "MU/USDC",    0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MU",    InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    HOODUSDC  = ["HOODUSDC",  "HOOD/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:HOOD",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.FINANCIALS]
    ORCLUSDC  = ["ORCLUSDC",  "ORCL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:ORCL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.INFORMATION_TECHNOLOGY]
    SPCXUSDC  = ["SPCXUSDC",  "SPCX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:SPCX",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5), ExposureGroup.COMMUNICATION]

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
