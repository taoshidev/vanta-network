# developer: Taoshi
"""TradePair enum and supporting types/constants.

Split out of vali_config.py so the trade-pair domain has its own module.
vali_config.py re-exports the public classes from here for backwards compatibility.
"""
from collections import defaultdict
from enum import Enum
from typing import NamedTuple


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

# Trade-pair id sets used by TradePair.is_blocked / is_flat_only.
FLAT_ONLY_TRADE_PAIR_IDS = {}
BLOCKED_TRADE_PAIR_IDS = {
    'SPX', 'DJI', 'NDX', 'VIX', 'FTSE', 'GDAXI',  # Indices
    'USDMXN',
    'PAXGUSDC',      # Gold; kept GOLDUSDC
    'BRENTOILUSDC',  # Oil; kept WTIOILUSDC
    'XAGUSD', 'XAUUSD',  # replaced with GOLDUSDC, SILVERUSDC
    'TONUSDC',  # Delisted from Hyperliquid
    'BTCUSD',
    'ETHUSD',
    'SOLUSD',
    'XRPUSD',
    'DOGEUSD',
    'ADAUSD',
    'TAOUSD',
    'HYPEUSD',
    'ZECUSD',
    'BCHUSD',
    'LINKUSD',
    'XMRUSD',
    'LTCUSD',
}


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


class TradePair(Enum):
    # Vanta Native Trade Pairs
    # crypto
    BTCUSD = ["BTCUSD", "BTC/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.MAJORS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ETHUSD = ["ETHUSD", "ETH/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.MAJORS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SOLUSD = ["SOLUSD", "SOL/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XRPUSD = ["XRPUSD", "XRP/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DOGEUSD = ["DOGEUSD", "DOGE/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADAUSD = ["ADAUSD", "ADA/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TAOUSD = ["TAOUSD", "TAO/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HYPEUSD = ["HYPEUSD", "HYPE/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZECUSD = ["ZECUSD", "ZEC/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BCHUSD = ["BCHUSD", "BCH/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LINKUSD = ["LINKUSD", "LINK/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XMRUSD = ["XMRUSD", "XMR/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LTCUSD = ["LTCUSD", "LTC/USD", 0.001, CRYPTO_MIN_LEVERAGE, CRYPTO_MAX_LEVERAGE, TradePairCategory.CRYPTO, CryptoSubcategory.ALTS, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]

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
    NVDA = ["NVDA", "NVDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MSFT = ["MSFT", "MSFT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AAPL = ["AAPL", "AAPL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AVGO = ["AVGO", "AVGO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TSM = ["TSM", "TSM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ORCL = ["ORCL", "ORCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMD = ["AMD", "AMD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MU = ["MU", "MU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CRM = ["CRM", "CRM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UBER = ["UBER", "UBER", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    # Financial Services (5)
    BRK_B = ["BRK_B", "BRK.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JPM = ["JPM", "JPM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    V = ["V", "V", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MA = ["MA", "MA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BAC = ["BAC", "BAC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    # Consumer Discretionary (5)
    AMZN = ["AMZN", "AMZN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TSLA = ["TSLA", "TSLA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HD = ["HD", "HD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BABA = ["BABA", "BABA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SBUX = ["SBUX", "SBUX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    # Communication Services (5)
    GOOGL = ["GOOGL", "GOOGL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    META = ["META", "META", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NFLX = ["NFLX", "NFLX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    APP = ["APP", "APP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    T = ["T", "T", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    # Spot single stocks matching Hyperliquid equity perps (7)
    COIN = ["COIN", "COIN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CRCL = ["CRCL", "CRCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MSTR = ["MSTR", "MSTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PLTR = ["PLTR", "PLTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SNDK = ["SNDK", "SNDK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INTC = ["INTC", "INTC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HOOD = ["HOOD", "HOOD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]

    # Russell 1000 stocks bulk-added by runnable/generate_equity_universe.py (additive: appends new
    # tickers, never touches existing). Per-pair fees/base literals here are hand-editable.
    A = ["A", "A", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AA = ["AA", "AA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AAL = ["AAL", "AAL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AAON = ["AAON", "AAON", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ABBV = ["ABBV", "ABBV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ABNB = ["ABNB", "ABNB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ABT = ["ABT", "ABT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ACGL = ["ACGL", "ACGL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ACHC = ["ACHC", "ACHC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ACI = ["ACI", "ACI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ACM = ["ACM", "ACM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ACN = ["ACN", "ACN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADBE = ["ADBE", "ADBE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADC = ["ADC", "ADC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADI = ["ADI", "ADI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADM = ["ADM", "ADM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADP = ["ADP", "ADP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADSK = ["ADSK", "ADSK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ADT = ["ADT", "ADT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AEE = ["AEE", "AEE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AEP = ["AEP", "AEP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AES = ["AES", "AES", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AFG = ["AFG", "AFG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AFL = ["AFL", "AFL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AFRM = ["AFRM", "AFRM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AGCO = ["AGCO", "AGCO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AGNC = ["AGNC", "AGNC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AGO = ["AGO", "AGO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AIG = ["AIG", "AIG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AIT = ["AIT", "AIT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AIZ = ["AIZ", "AIZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AJG = ["AJG", "AJG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AKAM = ["AKAM", "AKAM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALAB = ["ALAB", "ALAB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALB = ["ALB", "ALB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALGM = ["ALGM", "ALGM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALGN = ["ALGN", "ALGN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALK = ["ALK", "ALK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALL = ["ALL", "ALL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALLE = ["ALLE", "ALLE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALLY = ["ALLY", "ALLY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALNY = ["ALNY", "ALNY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ALSN = ["ALSN", "ALSN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AM = ["AM", "AM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMAT = ["AMAT", "AMAT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMCR = ["AMCR", "AMCR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AME = ["AME", "AME", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMG = ["AMG", "AMG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMGN = ["AMGN", "AMGN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMH = ["AMH", "AMH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMKR = ["AMKR", "AMKR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMP = ["AMP", "AMP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMT = ["AMT", "AMT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AMTM = ["AMTM", "AMTM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AN = ["AN", "AN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ANET = ["ANET", "ANET", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AON = ["AON", "AON", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AOS = ["AOS", "AOS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    APA = ["APA", "APA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    APD = ["APD", "APD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    APG = ["APG", "APG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    APH = ["APH", "APH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    APO = ["APO", "APO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    APPF = ["APPF", "APPF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    APTV = ["APTV", "APTV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AR = ["AR", "AR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ARE = ["ARE", "ARE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ARES = ["ARES", "ARES", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ARMK = ["ARMK", "ARMK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ARW = ["ARW", "ARW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AS = ["AS", "AS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ASH = ["ASH", "ASH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ASTS = ["ASTS", "ASTS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ATI = ["ATI", "ATI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ATO = ["ATO", "ATO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ATR = ["ATR", "ATR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AU = ["AU", "AU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AUR = ["AUR", "AUR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AVB = ["AVB", "AVB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AVT = ["AVT", "AVT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AVTR = ["AVTR", "AVTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AVY = ["AVY", "AVY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AWI = ["AWI", "AWI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AWK = ["AWK", "AWK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AXON = ["AXON", "AXON", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AXP = ["AXP", "AXP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AXS = ["AXS", "AXS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AXTA = ["AXTA", "AXTA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AYI = ["AYI", "AYI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    AZO = ["AZO", "AZO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BA = ["BA", "BA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BAH = ["BAH", "BAH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BALL = ["BALL", "BALL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BAM = ["BAM", "BAM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BAX = ["BAX", "BAX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BBWI = ["BBWI", "BBWI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BBY = ["BBY", "BBY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BC = ["BC", "BC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BDX = ["BDX", "BDX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BEN = ["BEN", "BEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BEPC = ["BEPC", "BEPC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BFAM = ["BFAM", "BFAM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BF_A = ["BF_A", "BF.A", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BF_B = ["BF_B", "BF.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BG = ["BG", "BG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BHF = ["BHF", "BHF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BIIB = ["BIIB", "BIIB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BILL = ["BILL", "BILL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BIO = ["BIO", "BIO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BIRK = ["BIRK", "BIRK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BJ = ["BJ", "BJ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BKNG = ["BKNG", "BKNG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BKR = ["BKR", "BKR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BLD = ["BLD", "BLD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BLDR = ["BLDR", "BLDR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BLK = ["BLK", "BLK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BLSH = ["BLSH", "BLSH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BMRN = ["BMRN", "BMRN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BMY = ["BMY", "BMY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BNY = ["BNY", "BNY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BOKF = ["BOKF", "BOKF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BPOP = ["BPOP", "BPOP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BR = ["BR", "BR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BRBR = ["BRBR", "BRBR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BRKR = ["BRKR", "BRKR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BRO = ["BRO", "BRO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BROS = ["BROS", "BROS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BRX = ["BRX", "BRX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BSX = ["BSX", "BSX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BSY = ["BSY", "BSY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BURL = ["BURL", "BURL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BWA = ["BWA", "BWA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BWXT = ["BWXT", "BWXT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BX = ["BX", "BX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BXP = ["BXP", "BXP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    BYD = ["BYD", "BYD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    C = ["C", "C", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CACC = ["CACC", "CACC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CACI = ["CACI", "CACI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CAG = ["CAG", "CAG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CAH = ["CAH", "CAH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CAI = ["CAI", "CAI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CAR = ["CAR", "CAR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CARR = ["CARR", "CARR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CART = ["CART", "CART", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CASY = ["CASY", "CASY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CAT = ["CAT", "CAT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CAVA = ["CAVA", "CAVA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CB = ["CB", "CB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CBC = ["CBC", "CBC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CBOE = ["CBOE", "CBOE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CBRE = ["CBRE", "CBRE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CBSH = ["CBSH", "CBSH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CCC = ["CCC", "CCC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CCI = ["CCI", "CCI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CCK = ["CCK", "CCK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CCL = ["CCL", "CCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CDNS = ["CDNS", "CDNS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CDW = ["CDW", "CDW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CE = ["CE", "CE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CEG = ["CEG", "CEG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CELH = ["CELH", "CELH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CERT = ["CERT", "CERT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CF = ["CF", "CF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CFG = ["CFG", "CFG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CFR = ["CFR", "CFR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CG = ["CG", "CG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CGNX = ["CGNX", "CGNX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CHD = ["CHD", "CHD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CHDN = ["CHDN", "CHDN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CHE = ["CHE", "CHE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CHH = ["CHH", "CHH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CHRD = ["CHRD", "CHRD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CHRW = ["CHRW", "CHRW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CHTR = ["CHTR", "CHTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CHWY = ["CHWY", "CHWY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CI = ["CI", "CI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CIEN = ["CIEN", "CIEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CINF = ["CINF", "CINF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CL = ["CL", "CL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CLF = ["CLF", "CLF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CLH = ["CLH", "CLH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CLVT = ["CLVT", "CLVT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CLX = ["CLX", "CLX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CMCSA = ["CMCSA", "CMCSA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CME = ["CME", "CME", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CMG = ["CMG", "CMG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CMI = ["CMI", "CMI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CMS = ["CMS", "CMS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CNA = ["CNA", "CNA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CNC = ["CNC", "CNC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CNH = ["CNH", "CNH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CNM = ["CNM", "CNM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CNP = ["CNP", "CNP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CNXC = ["CNXC", "CNXC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COF = ["COF", "COF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COHR = ["COHR", "COHR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COKE = ["COKE", "COKE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COLB = ["COLB", "COLB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COLD = ["COLD", "COLD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COLM = ["COLM", "COLM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COO = ["COO", "COO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COP = ["COP", "COP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COR = ["COR", "COR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CORT = ["CORT", "CORT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COST = ["COST", "COST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    COTY = ["COTY", "COTY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CPAY = ["CPAY", "CPAY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CPB = ["CPB", "CPB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CPNG = ["CPNG", "CPNG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CPRT = ["CPRT", "CPRT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CPT = ["CPT", "CPT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CR = ["CR", "CR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CRBG = ["CRBG", "CRBG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CRH = ["CRH", "CRH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CRL = ["CRL", "CRL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CROX = ["CROX", "CROX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CRS = ["CRS", "CRS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CRUS = ["CRUS", "CRUS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CRWD = ["CRWD", "CRWD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CSCO = ["CSCO", "CSCO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CSGP = ["CSGP", "CSGP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CSL = ["CSL", "CSL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CSX = ["CSX", "CSX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CTAS = ["CTAS", "CTAS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CTSH = ["CTSH", "CTSH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CTVA = ["CTVA", "CTVA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CUBE = ["CUBE", "CUBE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CUZ = ["CUZ", "CUZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CVNA = ["CVNA", "CVNA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CVS = ["CVS", "CVS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CVX = ["CVX", "CVX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CW = ["CW", "CW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CWEN = ["CWEN", "CWEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CXT = ["CXT", "CXT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    CZR = ["CZR", "CZR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    D = ["D", "D", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DAL = ["DAL", "DAL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DAR = ["DAR", "DAR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DASH = ["DASH", "DASH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DBX = ["DBX", "DBX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DCI = ["DCI", "DCI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DD = ["DD", "DD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DDOG = ["DDOG", "DDOG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DDS = ["DDS", "DDS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DE = ["DE", "DE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DECK = ["DECK", "DECK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DELL = ["DELL", "DELL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DG = ["DG", "DG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DGX = ["DGX", "DGX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DHI = ["DHI", "DHI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DHR = ["DHR", "DHR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DINO = ["DINO", "DINO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DIS = ["DIS", "DIS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DJT = ["DJT", "DJT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DKNG = ["DKNG", "DKNG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DKS = ["DKS", "DKS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DLB = ["DLB", "DLB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DLR = ["DLR", "DLR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DLTR = ["DLTR", "DLTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DOC = ["DOC", "DOC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DOCS = ["DOCS", "DOCS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DOCU = ["DOCU", "DOCU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DOV = ["DOV", "DOV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DOW = ["DOW", "DOW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DOX = ["DOX", "DOX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DPZ = ["DPZ", "DPZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DRI = ["DRI", "DRI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DRS = ["DRS", "DRS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DT = ["DT", "DT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DTE = ["DTE", "DTE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DTM = ["DTM", "DTM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DUK = ["DUK", "DUK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DUOL = ["DUOL", "DUOL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DV = ["DV", "DV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DVA = ["DVA", "DVA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DVN = ["DVN", "DVN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DXC = ["DXC", "DXC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    DXCM = ["DXCM", "DXCM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EA = ["EA", "EA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EBAY = ["EBAY", "EBAY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ECG = ["ECG", "ECG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ECL = ["ECL", "ECL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ED = ["ED", "ED", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EEFT = ["EEFT", "EEFT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EFX = ["EFX", "EFX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EG = ["EG", "EG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EGP = ["EGP", "EGP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EHC = ["EHC", "EHC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EIX = ["EIX", "EIX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EL = ["EL", "EL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ELAN = ["ELAN", "ELAN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ELF = ["ELF", "ELF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ELS = ["ELS", "ELS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ELV = ["ELV", "ELV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EME = ["EME", "EME", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EMN = ["EMN", "EMN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EMR = ["EMR", "EMR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ENPH = ["ENPH", "ENPH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ENTG = ["ENTG", "ENTG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EOG = ["EOG", "EOG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EPAM = ["EPAM", "EPAM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EPR = ["EPR", "EPR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EQH = ["EQH", "EQH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EQIX = ["EQIX", "EQIX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EQR = ["EQR", "EQR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EQT = ["EQT", "EQT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ES = ["ES", "ES", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ESAB = ["ESAB", "ESAB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ESI = ["ESI", "ESI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ESS = ["ESS", "ESS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ESTC = ["ESTC", "ESTC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ETN = ["ETN", "ETN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ETR = ["ETR", "ETR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ETSY = ["ETSY", "ETSY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EVR = ["EVR", "EVR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EVRG = ["EVRG", "EVRG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EW = ["EW", "EW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EWBC = ["EWBC", "EWBC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EXC = ["EXC", "EXC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EXE = ["EXE", "EXE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EXEL = ["EXEL", "EXEL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EXLS = ["EXLS", "EXLS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EXP = ["EXP", "EXP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EXPD = ["EXPD", "EXPD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EXPE = ["EXPE", "EXPE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    EXR = ["EXR", "EXR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    F = ["F", "F", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FAF = ["FAF", "FAF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FANG = ["FANG", "FANG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FAST = ["FAST", "FAST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FBIN = ["FBIN", "FBIN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FCN = ["FCN", "FCN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FCNCA = ["FCNCA", "FCNCA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FCX = ["FCX", "FCX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FDS = ["FDS", "FDS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FDX = ["FDX", "FDX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FDXF = ["FDXF", "FDXF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FE = ["FE", "FE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FERG = ["FERG", "FERG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FFIV = ["FFIV", "FFIV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FHB = ["FHB", "FHB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FHN = ["FHN", "FHN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FICO = ["FICO", "FICO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FIGR = ["FIGR", "FIGR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FIS = ["FIS", "FIS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FISV = ["FISV", "FISV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FITB = ["FITB", "FITB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FIVE = ["FIVE", "FIVE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FIX = ["FIX", "FIX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FLEX = ["FLEX", "FLEX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FLO = ["FLO", "FLO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FLS = ["FLS", "FLS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FLUT = ["FLUT", "FLUT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FMC = ["FMC", "FMC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FNB = ["FNB", "FNB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FND = ["FND", "FND", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FNF = ["FNF", "FNF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FOUR = ["FOUR", "FOUR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FOX = ["FOX", "FOX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FOXA = ["FOXA", "FOXA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FR = ["FR", "FR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FRHC = ["FRHC", "FRHC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FRMI = ["FRMI", "FRMI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FRPT = ["FRPT", "FRPT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FRT = ["FRT", "FRT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FSLR = ["FSLR", "FSLR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FTAI = ["FTAI", "FTAI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FTI = ["FTI", "FTI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FTNT = ["FTNT", "FTNT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FTV = ["FTV", "FTV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FWONA = ["FWONA", "FWONA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    FWONK = ["FWONK", "FWONK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    G = ["G", "G", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GAP = ["GAP", "GAP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GD = ["GD", "GD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GDDY = ["GDDY", "GDDY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GE = ["GE", "GE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GEHC = ["GEHC", "GEHC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GEN = ["GEN", "GEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GEV = ["GEV", "GEV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GFS = ["GFS", "GFS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GGG = ["GGG", "GGG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GILD = ["GILD", "GILD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GIS = ["GIS", "GIS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GL = ["GL", "GL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GLIBA = ["GLIBA", "GLIBA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GLIBK = ["GLIBK", "GLIBK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GLOB = ["GLOB", "GLOB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GLPI = ["GLPI", "GLPI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GLW = ["GLW", "GLW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GM = ["GM", "GM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GME = ["GME", "GME", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GMED = ["GMED", "GMED", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GNRC = ["GNRC", "GNRC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GNTX = ["GNTX", "GNTX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GOOG = ["GOOG", "GOOG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GPC = ["GPC", "GPC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GPK = ["GPK", "GPK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GPN = ["GPN", "GPN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GRMN = ["GRMN", "GRMN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GS = ["GS", "GS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GTES = ["GTES", "GTES", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GTLB = ["GTLB", "GTLB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GTM = ["GTM", "GTM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GWRE = ["GWRE", "GWRE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GWW = ["GWW", "GWW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    GXO = ["GXO", "GXO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    H = ["H", "H", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HAL = ["HAL", "HAL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HALO = ["HALO", "HALO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HAS = ["HAS", "HAS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HAYW = ["HAYW", "HAYW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HBAN = ["HBAN", "HBAN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HCA = ["HCA", "HCA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HEI = ["HEI", "HEI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HEI_A = ["HEI_A", "HEI.A", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HHH = ["HHH", "HHH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HIG = ["HIG", "HIG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HII = ["HII", "HII", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HIW = ["HIW", "HIW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HLI = ["HLI", "HLI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HLNE = ["HLNE", "HLNE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HLT = ["HLT", "HLT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HOG = ["HOG", "HOG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HON = ["HON", "HON", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HPE = ["HPE", "HPE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HPQ = ["HPQ", "HPQ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HR = ["HR", "HR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HRB = ["HRB", "HRB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HRL = ["HRL", "HRL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HSIC = ["HSIC", "HSIC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HST = ["HST", "HST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HSY = ["HSY", "HSY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HUBB = ["HUBB", "HUBB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HUBS = ["HUBS", "HUBS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HUM = ["HUM", "HUM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HUN = ["HUN", "HUN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HWM = ["HWM", "HWM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    HXL = ["HXL", "HXL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IBKR = ["IBKR", "IBKR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IBM = ["IBM", "IBM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ICE = ["ICE", "ICE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IDA = ["IDA", "IDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IDXX = ["IDXX", "IDXX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IEX = ["IEX", "IEX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IFF = ["IFF", "IFF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ILMN = ["ILMN", "ILMN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INCY = ["INCY", "INCY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INGM = ["INGM", "INGM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INGR = ["INGR", "INGR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INSM = ["INSM", "INSM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INSP = ["INSP", "INSP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INTU = ["INTU", "INTU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    INVH = ["INVH", "INVH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IONS = ["IONS", "IONS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IOT = ["IOT", "IOT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IP = ["IP", "IP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IPGP = ["IPGP", "IPGP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IQV = ["IQV", "IQV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IR = ["IR", "IR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IRDM = ["IRDM", "IRDM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IRM = ["IRM", "IRM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ISRG = ["ISRG", "ISRG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IT = ["IT", "IT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ITT = ["ITT", "ITT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ITW = ["ITW", "ITW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    IVZ = ["IVZ", "IVZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    J = ["J", "J", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JAZZ = ["JAZZ", "JAZZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JBHT = ["JBHT", "JBHT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JBL = ["JBL", "JBL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JCI = ["JCI", "JCI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JEF = ["JEF", "JEF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JHG = ["JHG", "JHG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JHX = ["JHX", "JHX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JKHY = ["JKHY", "JKHY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JLL = ["JLL", "JLL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    JNJ = ["JNJ", "JNJ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KBR = ["KBR", "KBR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KD = ["KD", "KD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KDP = ["KDP", "KDP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KEX = ["KEX", "KEX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KEY = ["KEY", "KEY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KEYS = ["KEYS", "KEYS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KHC = ["KHC", "KHC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KIM = ["KIM", "KIM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KKR = ["KKR", "KKR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KLAC = ["KLAC", "KLAC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KMB = ["KMB", "KMB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KMI = ["KMI", "KMI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KMPR = ["KMPR", "KMPR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KMX = ["KMX", "KMX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KNSL = ["KNSL", "KNSL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KNX = ["KNX", "KNX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KO = ["KO", "KO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KR = ["KR", "KR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KRC = ["KRC", "KRC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KRMN = ["KRMN", "KRMN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    KVUE = ["KVUE", "KVUE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    L = ["L", "L", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LAD = ["LAD", "LAD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LAMR = ["LAMR", "LAMR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LAZ = ["LAZ", "LAZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LBRDA = ["LBRDA", "LBRDA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LBRDK = ["LBRDK", "LBRDK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LBTYA = ["LBTYA", "LBTYA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LBTYK = ["LBTYK", "LBTYK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LCID = ["LCID", "LCID", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LDOS = ["LDOS", "LDOS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LEA = ["LEA", "LEA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LECO = ["LECO", "LECO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LEN = ["LEN", "LEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LEN_B = ["LEN_B", "LEN.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LFUS = ["LFUS", "LFUS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LH = ["LH", "LH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LHX = ["LHX", "LHX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LII = ["LII", "LII", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LIN = ["LIN", "LIN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LINE = ["LINE", "LINE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LITE = ["LITE", "LITE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LKQ = ["LKQ", "LKQ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LLY = ["LLY", "LLY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LLYVA = ["LLYVA", "LLYVA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LLYVK = ["LLYVK", "LLYVK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LMT = ["LMT", "LMT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LNC = ["LNC", "LNC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LNG = ["LNG", "LNG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LNT = ["LNT", "LNT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LOAR = ["LOAR", "LOAR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LOPE = ["LOPE", "LOPE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LOW = ["LOW", "LOW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LPLA = ["LPLA", "LPLA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LPX = ["LPX", "LPX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LRCX = ["LRCX", "LRCX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LSCC = ["LSCC", "LSCC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LSTR = ["LSTR", "LSTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LULU = ["LULU", "LULU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LUV = ["LUV", "LUV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LVS = ["LVS", "LVS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LW = ["LW", "LW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LYB = ["LYB", "LYB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LYFT = ["LYFT", "LYFT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    LYV = ["LYV", "LYV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    M = ["M", "M", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MAA = ["MAA", "MAA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MAN = ["MAN", "MAN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MANH = ["MANH", "MANH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MAR = ["MAR", "MAR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MAS = ["MAS", "MAS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MAT = ["MAT", "MAT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MCD = ["MCD", "MCD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MCHP = ["MCHP", "MCHP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MCK = ["MCK", "MCK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MCO = ["MCO", "MCO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MDB = ["MDB", "MDB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MDLN = ["MDLN", "MDLN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MDLZ = ["MDLZ", "MDLZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MDT = ["MDT", "MDT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MDU = ["MDU", "MDU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MEDP = ["MEDP", "MEDP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MET = ["MET", "MET", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MGM = ["MGM", "MGM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MHK = ["MHK", "MHK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MIDD = ["MIDD", "MIDD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MKC = ["MKC", "MKC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MKL = ["MKL", "MKL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MKSI = ["MKSI", "MKSI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MKTX = ["MKTX", "MKTX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MLI = ["MLI", "MLI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MLM = ["MLM", "MLM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MMM = ["MMM", "MMM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MNST = ["MNST", "MNST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MO = ["MO", "MO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MOH = ["MOH", "MOH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MORN = ["MORN", "MORN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MOS = ["MOS", "MOS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MP = ["MP", "MP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MPC = ["MPC", "MPC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MPT = ["MPT", "MPT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MPWR = ["MPWR", "MPWR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MRK = ["MRK", "MRK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MRNA = ["MRNA", "MRNA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MRP = ["MRP", "MRP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MRSH = ["MRSH", "MRSH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MRVL = ["MRVL", "MRVL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MS = ["MS", "MS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MSA = ["MSA", "MSA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MSCI = ["MSCI", "MSCI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MSGS = ["MSGS", "MSGS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MSI = ["MSI", "MSI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MSM = ["MSM", "MSM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MTB = ["MTB", "MTB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MTCH = ["MTCH", "MTCH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MTD = ["MTD", "MTD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MTDR = ["MTDR", "MTDR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MTG = ["MTG", "MTG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MTN = ["MTN", "MTN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MTSI = ["MTSI", "MTSI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MTZ = ["MTZ", "MTZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    MUSA = ["MUSA", "MUSA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NBIX = ["NBIX", "NBIX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NCLH = ["NCLH", "NCLH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NCNO = ["NCNO", "NCNO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NDAQ = ["NDAQ", "NDAQ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NDSN = ["NDSN", "NDSN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NEE = ["NEE", "NEE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NEM = ["NEM", "NEM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NET = ["NET", "NET", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NEU = ["NEU", "NEU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NFG = ["NFG", "NFG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NI = ["NI", "NI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NIQ = ["NIQ", "NIQ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NKE = ["NKE", "NKE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NLY = ["NLY", "NLY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NNN = ["NNN", "NNN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NOC = ["NOC", "NOC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NOV = ["NOV", "NOV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NOW = ["NOW", "NOW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NRG = ["NRG", "NRG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NSA = ["NSA", "NSA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NSC = ["NSC", "NSC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NTAP = ["NTAP", "NTAP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NTNX = ["NTNX", "NTNX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NTRA = ["NTRA", "NTRA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NTRS = ["NTRS", "NTRS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NU = ["NU", "NU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NUE = ["NUE", "NUE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NVR = ["NVR", "NVR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NVST = ["NVST", "NVST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NVT = ["NVT", "NVT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NWL = ["NWL", "NWL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NWS = ["NWS", "NWS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NWSA = ["NWSA", "NWSA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NXST = ["NXST", "NXST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    NYT = ["NYT", "NYT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    O = ["O", "O", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OC = ["OC", "OC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ODFL = ["ODFL", "ODFL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OGE = ["OGE", "OGE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OGN = ["OGN", "OGN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OHI = ["OHI", "OHI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OKE = ["OKE", "OKE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OKTA = ["OKTA", "OKTA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OLED = ["OLED", "OLED", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OLLI = ["OLLI", "OLLI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OLN = ["OLN", "OLN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OMC = ["OMC", "OMC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OMF = ["OMF", "OMF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ON = ["ON", "ON", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ONON = ["ONON", "ONON", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ONTO = ["ONTO", "ONTO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ORI = ["ORI", "ORI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ORLY = ["ORLY", "ORLY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OSK = ["OSK", "OSK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OTIS = ["OTIS", "OTIS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OVV = ["OVV", "OVV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OWL = ["OWL", "OWL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OXY = ["OXY", "OXY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    OZK = ["OZK", "OZK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    P = ["P", "P", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PAG = ["PAG", "PAG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PANW = ["PANW", "PANW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PATH = ["PATH", "PATH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PAYC = ["PAYC", "PAYC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PAYX = ["PAYX", "PAYX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PB = ["PB", "PB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PCAR = ["PCAR", "PCAR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PCG = ["PCG", "PCG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PCOR = ["PCOR", "PCOR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PCTY = ["PCTY", "PCTY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PEG = ["PEG", "PEG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PEGA = ["PEGA", "PEGA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PEN = ["PEN", "PEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PENN = ["PENN", "PENN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PEP = ["PEP", "PEP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PFE = ["PFE", "PFE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PFG = ["PFG", "PFG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PFGC = ["PFGC", "PFGC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PG = ["PG", "PG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PGR = ["PGR", "PGR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PH = ["PH", "PH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PHM = ["PHM", "PHM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PINS = ["PINS", "PINS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PK = ["PK", "PK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PKG = ["PKG", "PKG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PLD = ["PLD", "PLD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PLNT = ["PLNT", "PLNT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PM = ["PM", "PM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PNC = ["PNC", "PNC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PNFP = ["PNFP", "PNFP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PNR = ["PNR", "PNR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PNW = ["PNW", "PNW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PODD = ["PODD", "PODD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    POOL = ["POOL", "POOL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    POST = ["POST", "POST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PPC = ["PPC", "PPC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PPG = ["PPG", "PPG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PPL = ["PPL", "PPL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PPLI = ["PPLI", "PPLI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PR = ["PR", "PR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PRGO = ["PRGO", "PRGO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PRI = ["PRI", "PRI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PRMB = ["PRMB", "PRMB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PRU = ["PRU", "PRU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PSA = ["PSA", "PSA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PSN = ["PSN", "PSN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PSX = ["PSX", "PSX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PTC = ["PTC", "PTC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PVH = ["PVH", "PVH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PWR = ["PWR", "PWR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    PYPL = ["PYPL", "PYPL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    Q = ["Q", "Q", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    QCOM = ["QCOM", "QCOM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    QGEN = ["QGEN", "QGEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    QRVO = ["QRVO", "QRVO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    QS = ["QS", "QS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    QSR = ["QSR", "QSR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    QXO = ["QXO", "QXO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    R = ["R", "R", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RAL = ["RAL", "RAL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RARE = ["RARE", "RARE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RBA = ["RBA", "RBA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RBC = ["RBC", "RBC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RBLX = ["RBLX", "RBLX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RBRK = ["RBRK", "RBRK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RCL = ["RCL", "RCL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RDDT = ["RDDT", "RDDT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    REG = ["REG", "REG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    REGN = ["REGN", "REGN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    REXR = ["REXR", "REXR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    REYN = ["REYN", "REYN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RF = ["RF", "RF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RGA = ["RGA", "RGA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RGEN = ["RGEN", "RGEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RGLD = ["RGLD", "RGLD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RH = ["RH", "RH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RHI = ["RHI", "RHI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RITM = ["RITM", "RITM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RIVN = ["RIVN", "RIVN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RJF = ["RJF", "RJF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RKLB = ["RKLB", "RKLB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RKT = ["RKT", "RKT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RL = ["RL", "RL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RLI = ["RLI", "RLI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RMD = ["RMD", "RMD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RNG = ["RNG", "RNG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RNR = ["RNR", "RNR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ROIV = ["ROIV", "ROIV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ROK = ["ROK", "ROK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ROKU = ["ROKU", "ROKU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ROL = ["ROL", "ROL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ROP = ["ROP", "ROP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ROST = ["ROST", "ROST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RPM = ["RPM", "RPM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RPRX = ["RPRX", "RPRX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RRC = ["RRC", "RRC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RRX = ["RRX", "RRX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RS = ["RS", "RS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RSG = ["RSG", "RSG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RTX = ["RTX", "RTX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RVMD = ["RVMD", "RVMD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RVTY = ["RVTY", "RVTY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RYAN = ["RYAN", "RYAN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    RYN = ["RYN", "RYN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    S = ["S", "S", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SAIA = ["SAIA", "SAIA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SAIC = ["SAIC", "SAIC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SAIL = ["SAIL", "SAIL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SAM = ["SAM", "SAM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SARO = ["SARO", "SARO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SBAC = ["SBAC", "SBAC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SCCO = ["SCCO", "SCCO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SCHW = ["SCHW", "SCHW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SCI = ["SCI", "SCI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SEB = ["SEB", "SEB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SEIC = ["SEIC", "SEIC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SF = ["SF", "SF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SFD = ["SFD", "SFD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SFM = ["SFM", "SFM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SGI = ["SGI", "SGI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SHC = ["SHC", "SHC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SHW = ["SHW", "SHW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SIRI = ["SIRI", "SIRI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SITE = ["SITE", "SITE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SJM = ["SJM", "SJM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SLB = ["SLB", "SLB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SLGN = ["SLGN", "SLGN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SLM = ["SLM", "SLM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SMCI = ["SMCI", "SMCI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SMG = ["SMG", "SMG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SMMT = ["SMMT", "SMMT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SN = ["SN", "SN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SNA = ["SNA", "SNA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SNDR = ["SNDR", "SNDR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SNOW = ["SNOW", "SNOW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SNPS = ["SNPS", "SNPS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SNX = ["SNX", "SNX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SO = ["SO", "SO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SOFI = ["SOFI", "SOFI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SOLS = ["SOLS", "SOLS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SOLV = ["SOLV", "SOLV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SON = ["SON", "SON", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SPG = ["SPG", "SPG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SPGI = ["SPGI", "SPGI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SPOT = ["SPOT", "SPOT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SRE = ["SRE", "SRE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SRPT = ["SRPT", "SRPT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SSB = ["SSB", "SSB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SSD = ["SSD", "SSD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SSNC = ["SSNC", "SSNC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ST = ["ST", "ST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    STAG = ["STAG", "STAG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    STE = ["STE", "STE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    STLD = ["STLD", "STLD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    STT = ["STT", "STT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    STWD = ["STWD", "STWD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    STZ = ["STZ", "STZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SUI = ["SUI", "SUI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SW = ["SW", "SW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SWK = ["SWK", "SWK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SWKS = ["SWKS", "SWKS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SYF = ["SYF", "SYF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SYK = ["SYK", "SYK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    SYY = ["SYY", "SYY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TAP = ["TAP", "TAP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TDC = ["TDC", "TDC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TDG = ["TDG", "TDG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TDY = ["TDY", "TDY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TEAM = ["TEAM", "TEAM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TECH = ["TECH", "TECH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TEM = ["TEM", "TEM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TER = ["TER", "TER", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TFC = ["TFC", "TFC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TFSL = ["TFSL", "TFSL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TFX = ["TFX", "TFX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TGT = ["TGT", "TGT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    THC = ["THC", "THC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    THG = ["THG", "THG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    THO = ["THO", "THO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TIGO = ["TIGO", "TIGO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TJX = ["TJX", "TJX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TKO = ["TKO", "TKO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TKR = ["TKR", "TKR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TLN = ["TLN", "TLN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TMO = ["TMO", "TMO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TMUS = ["TMUS", "TMUS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TNL = ["TNL", "TNL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TOL = ["TOL", "TOL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TOST = ["TOST", "TOST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TPG = ["TPG", "TPG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TPL = ["TPL", "TPL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TPR = ["TPR", "TPR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TREX = ["TREX", "TREX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TRGP = ["TRGP", "TRGP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TRMB = ["TRMB", "TRMB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TROW = ["TROW", "TROW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TRU = ["TRU", "TRU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TRV = ["TRV", "TRV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TSCO = ["TSCO", "TSCO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TSN = ["TSN", "TSN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TT = ["TT", "TT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TTC = ["TTC", "TTC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TTD = ["TTD", "TTD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TTEK = ["TTEK", "TTEK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TTWO = ["TTWO", "TTWO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TW = ["TW", "TW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TWLO = ["TWLO", "TWLO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TXN = ["TXN", "TXN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TXRH = ["TXRH", "TXRH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TXT = ["TXT", "TXT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    TYL = ["TYL", "TYL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    U = ["U", "U", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UA = ["UA", "UA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UAA = ["UAA", "UAA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UAL = ["UAL", "UAL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UDR = ["UDR", "UDR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UGI = ["UGI", "UGI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UHAL = ["UHAL", "UHAL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UHAL_B = ["UHAL_B", "UHAL.B", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UHS = ["UHS", "UHS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UI = ["UI", "UI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ULTA = ["ULTA", "ULTA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UNH = ["UNH", "UNH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UNM = ["UNM", "UNM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UNP = ["UNP", "UNP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UPS = ["UPS", "UPS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    URI = ["URI", "URI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    USB = ["USB", "USB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    USFD = ["USFD", "USFD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UTHR = ["UTHR", "UTHR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    UWMC = ["UWMC", "UWMC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VEEV = ["VEEV", "VEEV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VFC = ["VFC", "VFC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VGNT = ["VGNT", "VGNT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VICI = ["VICI", "VICI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VIK = ["VIK", "VIK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VIRT = ["VIRT", "VIRT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VKTX = ["VKTX", "VKTX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VLO = ["VLO", "VLO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VLTO = ["VLTO", "VLTO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VMC = ["VMC", "VMC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VMI = ["VMI", "VMI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VNO = ["VNO", "VNO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VNOM = ["VNOM", "VNOM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VNT = ["VNT", "VNT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VOYA = ["VOYA", "VOYA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VRSK = ["VRSK", "VRSK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VRSN = ["VRSN", "VRSN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VRT = ["VRT", "VRT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VRTX = ["VRTX", "VRTX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VSNT = ["VSNT", "VSNT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VST = ["VST", "VST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VTR = ["VTR", "VTR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VTRS = ["VTRS", "VTRS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VVV = ["VVV", "VVV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VZ = ["VZ", "VZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    W = ["W", "W", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WAB = ["WAB", "WAB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WAL = ["WAL", "WAL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WAT = ["WAT", "WAT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WBD = ["WBD", "WBD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WBS = ["WBS", "WBS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WCC = ["WCC", "WCC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WDAY = ["WDAY", "WDAY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WDC = ["WDC", "WDC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WEC = ["WEC", "WEC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WELL = ["WELL", "WELL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WEN = ["WEN", "WEN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WEX = ["WEX", "WEX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WFC = ["WFC", "WFC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WFRD = ["WFRD", "WFRD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WH = ["WH", "WH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WHR = ["WHR", "WHR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WING = ["WING", "WING", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WLK = ["WLK", "WLK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WM = ["WM", "WM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WMB = ["WMB", "WMB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WMS = ["WMS", "WMS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WMT = ["WMT", "WMT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WPC = ["WPC", "WPC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WRB = ["WRB", "WRB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WSC = ["WSC", "WSC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WSM = ["WSM", "WSM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WSO = ["WSO", "WSO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WST = ["WST", "WST", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WTFC = ["WTFC", "WTFC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WTM = ["WTM", "WTM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WTRG = ["WTRG", "WTRG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WTW = ["WTW", "WTW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WU = ["WU", "WU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WWD = ["WWD", "WWD", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WY = ["WY", "WY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    WYNN = ["WYNN", "WYNN", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XEL = ["XEL", "XEL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XOM = ["XOM", "XOM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XP = ["XP", "XP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XPO = ["XPO", "XPO", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XRAY = ["XRAY", "XRAY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XYL = ["XYL", "XYL", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XYZ = ["XYZ", "XYZ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    YETI = ["YETI", "YETI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    YUM = ["YUM", "YUM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    Z = ["Z", "Z", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZBH = ["ZBH", "ZBH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZBRA = ["ZBRA", "ZBRA", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZG = ["ZG", "ZG", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZION = ["ZION", "ZION", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZM = ["ZM", "ZM", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZS = ["ZS", "ZS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    ZTS = ["ZTS", "ZTS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]

    # Equities - Sector ETFs (22)
    XLK = ["XLK", "XLK", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VGT = ["VGT", "VGT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLF = ["XLF", "XLF", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VFH = ["VFH", "VFH", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLY = ["XLY", "XLY", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VCR = ["VCR", "VCR", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLC = ["XLC", "XLC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VOX = ["VOX", "VOX", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLV = ["XLV", "XLV", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VHT = ["VHT", "VHT", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLI = ["XLI", "XLI", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VIS = ["VIS", "VIS", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLP = ["XLP", "XLP", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VDC = ["VDC", "VDC", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLE = ["XLE", "XLE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VDE = ["VDE", "VDE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLB = ["XLB", "XLB", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VAW = ["VAW", "VAW", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLU = ["XLU", "XLU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VPU = ["VPU", "VPU", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    XLRE = ["XLRE", "XLRE", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]
    VNQ = ["VNQ", "VNQ", 0.00009, EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage(0.5)]

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
    SPX = ["SPX", "SPX", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    DJI = ["DJI", "DJI", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    NDX = ["NDX", "NDX", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    VIX = ["VIX", "VIX", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    FTSE = ["FTSE", "FTSE", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]
    GDAXI = ["GDAXI", "GDAXI", 0.00009, INDICES_MIN_LEVERAGE, INDICES_MAX_LEVERAGE, TradePairCategory.INDICES, InstrumentType.SPOT, SubaccountTierBaseLeverage(2.5)]

    # Hyperliquid Trade Pairs (USDC-quoted, src=HYPERLIQUID)
    # Crypto perp futures
    BTCUSDC   = ["BTCUSDC",   "BTC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ETHUSDC   = ["ETHUSDC",   "ETH/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    SOLUSDC   = ["SOLUSDC",   "SOL/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    BNBUSDC   = ["BNBUSDC",   "BNB/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    XRPUSDC   = ["XRPUSDC",   "XRP/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    DOGEUSDC  = ["DOGEUSDC",  "DOGE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ADAUSDC   = ["ADAUSDC",   "ADA/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    AVAXUSDC  = ["AVAXUSDC",  "AVAX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    LINKUSDC  = ["LINKUSDC",  "LINK/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    DOTUSDC   = ["DOTUSDC",   "DOT/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TONUSDC   = ["TONUSDC",   "TON/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TRXUSDC   = ["TRXUSDC",   "TRX/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    LTCUSDC   = ["LTCUSDC",   "LTC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TAOUSDC   = ["TAOUSDC",   "TAO/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    SUIUSDC   = ["SUIUSDC",   "SUI/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ARBUSDC   = ["ARBUSDC",   "ARB/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    NEARUSDC  = ["NEARUSDC",  "NEAR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ALGOUSDC  = ["ALGOUSDC",  "ALGO/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ASTERUSDC = ["ASTERUSDC", "ASTER/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    UNIUSDC   = ["UNIUSDC",   "UNI/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    AAVEUSDC  = ["AAVEUSDC",  "AAVE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    CRVUSDC   = ["CRVUSDC",   "CRV/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    HYPEUSDC  = ["HYPEUSDC",  "HYPE/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    XMRUSDC   = ["XMRUSDC",   "XMR/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ZECUSDC   = ["ZECUSDC",   "ZEC/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    PAXGUSDC  = ["PAXGUSDC",  "PAXG/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ENAUSDC   = ["ENAUSDC",   "ENA/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ZROUSDC   = ["ZROUSDC",   "ZRO/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    WLDUSDC   = ["WLDUSDC",   "WLD/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    PUMPUSDC  = ["PUMPUSDC",  "PUMP/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    KPEPEUSDC = ["kPEPEUSDC", "kPEPE/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.CRYPTO,      None, TradePairSource.HYPERLIQUID, InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]

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
    NVDAUSDC  = ["NVDAUSDC",  "NVDA/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:NVDA",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    AAPLUSDC  = ["AAPLUSDC",  "AAPL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AAPL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TSLAUSDC  = ["TSLAUSDC",  "TSLA/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:TSLA",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    MSFTUSDC  = ["MSFTUSDC",  "MSFT/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MSFT",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    AMZNUSDC  = ["AMZNUSDC",  "AMZN/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AMZN",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    GOOGLUSDC = ["GOOGLUSDC", "GOOGL/USDC", 0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:GOOGL", InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    METAUSDC  = ["METAUSDC",  "META/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:META",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    COINUSDC  = ["COINUSDC",  "COIN/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:COIN",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    CRCLUSDC  = ["CRCLUSDC",  "CRCL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:CRCL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    MSTRUSDC  = ["MSTRUSDC",  "MSTR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MSTR",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    PLTRUSDC  = ["PLTRUSDC",  "PLTR/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:PLTR",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    AMDUSDC   = ["AMDUSDC",   "AMD/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:AMD",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    TSMUSDC   = ["TSMUSDC",   "TSM/USDC",   0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:TSM",   InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    NFLXUSDC  = ["NFLXUSDC",  "NFLX/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:NFLX",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    SNDKUSDC  = ["SNDKUSDC",  "SNDK/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:SNDK",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    INTCUSDC  = ["INTCUSDC",  "INTC/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:INTC",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    MUUSDC    = ["MUUSDC",    "MU/USDC",    0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:MU",    InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    HOODUSDC  = ["HOODUSDC",  "HOOD/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:HOOD",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]
    ORCLUSDC  = ["ORCLUSDC",  "ORCL/USDC",  0.001, HS_MIN_LEVERAGE, HS_MAX_LEVERAGE, TradePairCategory.EQUITIES, None, TradePairSource.HYPERLIQUID, "xyz:ORCL",  InstrumentType.PERP, SubaccountTierBaseLeverage(0.5)]

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
        return str(self.__json__())


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
