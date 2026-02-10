# developer: Taoshi
from datetime import datetime, timezone
import os
import math
from collections import defaultdict
from enum import Enum

from meta import load_version

BASE_DIR = base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta_dict = load_version(os.path.join(base_directory, "meta", "meta.json"))
if meta_dict is None:
    #  Databricks
    print('Unable to load meta_dict. This is expected if running on Databricks.')
    meta_version = "x.x.x"
else:
    meta_version = meta_dict.get("subnet_version", "x.x.x")

class RPCConnectionMode(int, Enum):
    """
    Connection mode for RPC clients/servers.

    LOCAL: Direct mode - bypass RPC, use set_direct_server() for in-process communication.
           Use this for tests that need to verify logic without RPC overhead.
    RPC: Normal RPC mode - connect via network.
           Use this for production and integration tests that need full RPC behavior.

    Usage:
        # Test without RPC (fastest, no network)
        client = MyClient(connection_mode=RPCConnectionMode.LOCAL)
        client.set_direct_server(server_instance)

        # Test with real RPC (like production)
        server = MyServer(connection_mode=RPCConnectionMode.RPC)  # Starts RPC server
        client = MyClient(connection_mode=RPCConnectionMode.RPC)  # Connects via RPC
    """
    LOCAL = 0   # Direct mode - bypass RPC, use set_direct_server()
    RPC = 1     # Normal RPC mode - connect via network


class TradePairCategory(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    INDICES = "indices"
    EQUITIES = "equities"


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


def _TradePair_Lookup() -> dict[str, TradePairCategory]:
    """
    Walk through every *concrete* subclass of TradePairSubcategory,
    collect their members, and map the member's *value* (your string)
    to its TradePairCategory.
    """
    mapping: dict[str, TradePairCategory] = {}

    # subclasses() finds *direct* children; recurse for grand‑children.
    def _walk(cls):
        for subcls in cls.__subclasses__():
            if issubclass(subcls, Enum):
                _walk(subcls)
                # subcls is itself an Enum: add all its members
                for member in subcls:
                    mapping[member.value] = member.asset_class

    _walk(TradePairSubcategory)
    return mapping

class InterpolatedValueFromDate():
    """
    Dynamic value based on dates. Used for setting configs in the future.
    """
    def __init__(self, start_date: str, *, low: int=None, high:int=None, interval: int, increment: int, target: int):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.low = low
        self.high = high
        self.interval = interval
        self.increment = increment
        self.target = target

    def value(self):
        days_since_start = (datetime.now(tz=timezone.utc) - self.start_date).days
        intervals = max(0, days_since_start // self.interval)

        if self.low is not None:
            new_n = self.low + abs(self.increment) * intervals
            return min(self.target, new_n)
        else:
            new_n = self.high - abs(self.increment) * intervals
            return max(self.target, new_n)

class ValiConfig:
    # versioning
    VERSION = meta_version

    # minimum required vanta-cli version
    VANTA_CLI_MINIMUM_VERSION = "1.0.5"

    DAYS_IN_YEAR_CRYPTO = 365  # annualization factor
    DAYS_IN_YEAR_FOREX = 252
    DAYS_IN_YEAR_EQUITIES = 252

    # Development hotkey for testing
    DEVELOPMENT_HOTKEY = "DEVELOPMENT"

    # RPC Service Configuration
    # Centralized port and service name definitions to avoid conflicts and inconsistencies
    # All RPC services are defined here to prevent port conflicts and ensure consistent authkey generation

    # Core Manager Services
    RPC_LIVEPRICEFETCHER_PORT = 50000
    RPC_LIVEPRICEFETCHER_SERVICE_NAME = "LivePriceFetcherServer"

    RPC_LIMITORDERMANAGER_PORT = 50001
    RPC_LIMITORDERMANAGER_SERVICE_NAME = "LimitOrderServer"

    RPC_POSITIONMANAGER_PORT = 50002
    RPC_POSITIONMANAGER_SERVICE_NAME = "PositionManagerServer"

    RPC_CHALLENGEPERIOD_PORT = 50003
    RPC_CHALLENGEPERIOD_SERVICE_NAME = "ChallengePeriodServer"

    RPC_ELIMINATION_PORT = 50004
    RPC_ELIMINATION_SERVICE_NAME = "EliminationServer"

    RPC_METAGRAPH_PORT = 50005
    RPC_METAGRAPH_SERVICE_NAME = "MetagraphServer"

    RPC_MINERSTATS_PORT = 50006
    RPC_MINERSTATS_SERVICE_NAME = "MinerStatsServer"

    RPC_COREOUTPUTS_PORT = 50007
    RPC_COREOUTPUTS_SERVICE_NAME = "CoreOutputsServer"

    # Utility Services
    RPC_POSITIONLOCK_PORT = 50008
    RPC_POSITIONLOCK_SERVICE_NAME = "PositionLockServer"

    RPC_DEBTLEDGER_PORT = 50009
    RPC_DEBTLEDGER_SERVICE_NAME = "DebtLedgerServer"

    RPC_ASSETSELECTION_PORT = 50010
    RPC_ASSETSELECTION_SERVICE_NAME = "AssetSelectionServer"

    RPC_CONTRACTMANAGER_PORT = 50011
    RPC_CONTRACTMANAGER_SERVICE_NAME = "ValidatorContractServer"

    RPC_MINERSTATISTICS_PORT = 50012
    RPC_MINERSTATISTICS_SERVICE_NAME = "MinerStatisticsServer"

    RPC_REQUESTCORE_PORT = 50013
    RPC_REQUESTCORE_SERVICE_NAME = "RequestCoreServer"

    RPC_WEBSOCKET_NOTIFIER_PORT = 50014
    RPC_WEBSOCKET_NOTIFIER_SERVICE_NAME = "WebSocketNotifierServer"

    RPC_WEIGHT_SETTER_PORT = 50015
    RPC_WEIGHT_SETTER_SERVICE_NAME = "WeightSetterServer"

    RPC_PERFLEDGER_PORT = 50016
    RPC_PERFLEDGER_SERVICE_NAME = "PerfLedgerServer"

    RPC_PLAGIARISM_PORT = 50017
    RPC_PLAGIARISM_SERVICE_NAME = "PlagiarismServer"

    RPC_PLAGIARISM_DETECTOR_PORT = 50018
    RPC_PLAGIARISM_DETECTOR_SERVICE_NAME = "PlagiarismDetectorServer"

    RPC_COMMONDATA_PORT = 50019
    RPC_COMMONDATA_SERVICE_NAME = "CommonDataServer"

    RPC_MDDCHECKER_PORT = 50020
    RPC_MDDCHECKER_SERVICE_NAME = "MDDCheckerServer"

    RPC_WEIGHT_CALCULATOR_PORT = 50021
    RPC_WEIGHT_CALCULATOR_SERVICE_NAME = "WeightCalculatorServer"

    RPC_REST_SERVER_PORT = 50022
    RPC_REST_SERVER_SERVICE_NAME = "VantaRestServer"

    RPC_MINERACCOUNT_PORT = 50023
    RPC_MINERACCOUNT_SERVICE_NAME = "MinerAccountServer"

    RPC_ENTITY_PORT = 50024
    RPC_ENTITY_SERVICE_NAME = "EntityServer"

    # Public API Configuration (well-known network endpoints)
    REST_API_HOST = "127.0.0.1"
    REST_API_PORT = 48888

    VANTA_WEBSOCKET_HOST = "localhost"
    VANTA_WEBSOCKET_PORT = 8765

    @staticmethod
    def get_rpc_authkey(service_name: str, port: int) -> bytes:
        """
        Generate RPC authkey for a service.

        Args:
            service_name: Service name (e.g., "ChallengePeriodManagerServer")
            port: Port number (e.g., 50003)

        Returns:
            bytes: 32-byte authkey for RPC authentication
        """
        import hashlib
        return hashlib.sha256(f"{service_name}_{port}".encode()).digest()[:32]

    # Min number of trading days required for scoring
    STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL = 60
    STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR = 7

    # Dynamic minimum days calculation - use Nth longest participating miner as threshold
    DYNAMIC_MIN_DAYS_NUM_MINERS = 20

    # Market-specific configurations
    ANNUAL_RISK_FREE_PERCENTAGE = 3.89  # From tbill rates
    ANNUAL_RISK_FREE_DECIMAL = ANNUAL_RISK_FREE_PERCENTAGE / 100
    DAILY_LOG_RISK_FREE_RATE_CRYPTO = math.log(1 + ANNUAL_RISK_FREE_DECIMAL) / DAYS_IN_YEAR_CRYPTO
    DAILY_LOG_RISK_FREE_RATE_FOREX = math.log(1 + ANNUAL_RISK_FREE_DECIMAL) / DAYS_IN_YEAR_FOREX
    MS_RISK_FREE_RATE = math.log(1 + ANNUAL_RISK_FREE_PERCENTAGE / 100) / (365 * 24 * 60 * 60 * 1000)

    # Asset Class Breakdown - defines the total emission for each asset class
    CATEGORY_LOOKUP: dict[str, TradePairCategory] = _TradePair_Lookup()
    ASSET_CLASS_BREAKDOWN = {
        TradePairCategory.CRYPTO: {
            "emission": 0.334,  # Total emission for crypto
            "days_in_year": DAYS_IN_YEAR_CRYPTO,
        },
        # These are based on margin requirements on brokerage accounts
        TradePairCategory.FOREX: {
            "emission": 0.333,  # Total emission for forex
            "days_in_year": DAYS_IN_YEAR_FOREX,
        },
        TradePairCategory.EQUITIES: {
            "emission": 0.333,  # Total emission for equities
            "days_in_year": DAYS_IN_YEAR_CRYPTO,
        },
    }

    # Time Configurations
    TARGET_CHECKPOINT_DURATION_MS = 1000 * 60 * 60 * 12  # 12 hours
    DAILY_MS = 1000 * 60 * 60 * 24  # 1 day
    DAILY_CHECKPOINTS = DAILY_MS // TARGET_CHECKPOINT_DURATION_MS  # 2 checkpoints per day

    # Set the target ledger window in days directly
    TARGET_LEDGER_WINDOW_DAYS = 120
    TARGET_LEDGER_WINDOW_MS = TARGET_LEDGER_WINDOW_DAYS * DAILY_MS
    # TARGET_LEDGER_N_CHECKPOINTS = TARGET_LEDGER_WINDOW_MS // TARGET_CHECKPOINT_DURATION_MS  # 180 checkpoints
    WEIGHTED_AVERAGE_DECAY_RATE = 0.075
    WEIGHTED_AVERAGE_DECAY_MIN = 0.15
    WEIGHTED_AVERAGE_DECAY_MAX = 1.0

    # Decay min specific for daily average PnL calculations
    WEIGHTED_AVERAGE_DECAY_MIN_PNL = 0.045 # Results in most recent 30 days having 70% weight

    POSITIONAL_EQUIVALENCE_WINDOW_MS = 1000 * 60 * 60 * 24  # 1 day

    SET_WEIGHT_REFRESH_TIME_MS = 60 * 5 * 1000  # 5 minutes
    SET_WEIGHT_LOOKBACK_RANGE_DAYS = TARGET_LEDGER_WINDOW_DAYS

    # Fees take into account exiting and entering a position, liquidity, and futures fees
    PERF_LEDGER_REFRESH_TIME_MS = 1000 * 60 * 5  # minutes
    CHALLENGE_PERIOD_REFRESH_TIME_MS = 1000 * 60 * 5  # minutes
    MDD_CHECK_REFRESH_TIME_MS = 60 * 1000  # 60 seconds
    PRICE_SOURCE_COMPACTING_SLEEP_INTERVAL_SECONDS = 60 * 60 * 12 # 12 hours

    # Positional Leverage limits
    CRYPTO_MIN_LEVERAGE = 0.01
    CRYPTO_MAX_LEVERAGE = 2.5
    FOREX_MIN_LEVERAGE = 0.1
    FOREX_MAX_LEVERAGE = 10
    INDICES_MIN_LEVERAGE = 0.1
    INDICES_MAX_LEVERAGE = 5
    EQUITIES_MIN_LEVERAGE = 0.1
    EQUITIES_MAX_LEVERAGE = 2

    MAX_DAILY_DRAWDOWN = 0.95  # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9  # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_TOTAL_DRAWDOWN_V2 = 0.95
    MAX_ORDERS_PER_POSITION = 100
    ORDER_COOLDOWN_MS = 10000  # 10 seconds
    ORDER_MIN_LEVERAGE = 0.001
    ORDER_MAX_LEVERAGE = 500

    # Controls how much history to store for price data which is used in retroactive updates
    RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS = 300000 # 5 minutes

    # Risk Profiling
    RISK_PROFILING_STEPS_MIN_LEVERAGE = min(CRYPTO_MIN_LEVERAGE, FOREX_MIN_LEVERAGE, INDICES_MIN_LEVERAGE, EQUITIES_MIN_LEVERAGE)
    RISK_PROFILING_STEPS_CRITERIA = 3
    RISK_PROFILING_MONOTONIC_CRITERIA = 2
    RISK_PROFILING_MARGIN_CRITERIA = 0.5
    RISK_PROFILING_LEVERAGE_ADVANCE = 1.5
    RISK_PROFILING_SCOPING_MECHANIC = 100
    RISK_PROFILING_SIGMOID_SHIFT = 1.2
    RISK_PROFILING_SIGMOID_SPREAD = 4
    # RISK_PROFILING_TIME_DECAY = 5
    # RISK_PROFILING_TIME_CYCLE = POSITIONAL_EQUIVALENCE_WINDOW_MS
    RISK_PROFILING_TIME_CRITERIA = 0.185  # threshold for the normalized error of a position’s order time intervals

    PLAGIARISM_MATCHING_TIME_RESOLUTION_MS = 60 * 1000 * 2  # 2 minutes
    PLAGIARISM_MAX_LAGS = 60
    PLAGIARISM_LOOKBACK_RANGE_MS = 10 * 24 * 60 * 60 * 1000  # 10 days
    PLAGIARISM_FOLLOWER_TIMELAG_THRESHOLD = 1.0005
    PLAGIARISM_FOLLOWER_SIMILARITY_THRESHOLD = 0.75
    PLAGIARISM_REPORTING_THRESHOLD = 0.8
    PLAGIARISM_REFRESH_TIME_MS = 1000 * 60 * 60 * 24 # 1 day
    PLAGIARISM_ORDER_TIME_WINDOW_MS = 1000 * 60 * 60 * 12
    PLAGIARISM_MINIMUM_FOLLOW_MS = 1000 * 10 # Minimum follow time of 10 seconds for each order

    EPSILON = 1e-6
    RETURN_SHORT_LOOKBACK_TIME_MS = 5 * 24 * 60 * 60 * 1000  # 5 days
    RETURN_SHORT_LOOKBACK_LEDGER_WINDOWS = RETURN_SHORT_LOOKBACK_TIME_MS // TARGET_CHECKPOINT_DURATION_MS


    MINIMUM_POSITION_DURATION_MS = 1 * 60 * 1000  # 1 minutes

    SHORT_LOOKBACK_WINDOW = 7 * DAILY_CHECKPOINTS

    # Scoring weights
    SCORING_OMEGA_WEIGHT = 0.0
    SCORING_SHARPE_WEIGHT = 0.0
    SCORING_SORTINO_WEIGHT = 0.0
    SCORING_STATISTICAL_CONFIDENCE_WEIGHT = 0.0
    SCORING_CALMAR_WEIGHT = 0.0
    SCORING_RETURN_WEIGHT = 0.0
    SCORING_PNL_WEIGHT = 1.0

    # Scoring hyperparameters
    OMEGA_LOSS_MINIMUM = 0.01   # Equivalent to 1% loss
    OMEGA_NOCONFIDENCE_VALUE = 0.0
    SHARPE_STDDEV_MINIMUM = 0.01  # Equivalent to 1% standard deviation
    SHARPE_NOCONFIDENCE_VALUE = -100
    SORTINO_DOWNSIDE_MINIMUM = 0.01  # Equivalent to 1% standard deviation
    SORTINO_NOCONFIDENCE_VALUE = -100
    STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE = -100
    CALMAR_NOCONFIDENCE_VALUE = -100
    PNL_NOCONFIDENCE_VALUE = 0

    # MDD penalty calculation
    APPROXIMATE_DRAWDOWN_PERCENTILE = 0.75
    DRAWDOWN_UPPER_SCALING = 5
    DRAWDOWN_MAXVALUE_PERCENTAGE = 10
    DRAWDOWN_MINVALUE_PERCENTAGE = 0.5

    # Risk Adjusted Performance Penalty
    CRYPTO_RAT = {'sharpe': 1.0, 'sortino': 1.0, 'calmar': 2.0, 'omega': 1.4}
    FOREX_RAT = {'sharpe': 0.5, 'sortino': 0.5, 'calmar': 2.0, 'omega': 1.2}

    # Maximum metric value for capping individual metrics in RAS calculation
    RISK_ADJUSTED_MAX_METRIC_VALUE = 10

    # Sigmoid parameters for risk-adjusted performance penalty (range: 0.2 to 1.0)
    RISK_ADJUSTED_SIGMOID_SHIFT = 0.6
    RISK_ADJUSTED_SIGMOID_SPREAD = -14
    RISK_ADJUSTED_PERFORMANCE_PENALTY_MIN = 0.2

    # Challenge period
    CHALLENGE_PERIOD_MIN_WEIGHT = 1.5e-05  # essentially nothing
    CHALLENGE_PERIOD_MAX_WEIGHT = 2.4e-05
    CHALLENGE_PERIOD_MINIMUM_DAYS = 61
    CHALLENGE_PERIOD_MAXIMUM_DAYS = 90
    CHALLENGE_PERIOD_MAXIMUM_MS = CHALLENGE_PERIOD_MAXIMUM_DAYS * DAILY_MS
    CHALLENGE_PERIOD_PERCENTILE_THRESHOLD = 0.75 # miners must pass 75th percentile to enter the main competition

    PROBATION_MAXIMUM_DAYS = 60
    PROBATION_MAXIMUM_MS = PROBATION_MAXIMUM_DAYS * DAILY_MS

    PROMOTION_THRESHOLD_RANK = 25 # Number of MAINCOMP miners per asset class

    # Plagiarism
    ORDER_SIMILARITY_WINDOW_MS = 60000 * 60 * 24
    MINER_COPYING_WEIGHT = 0.01
    MAX_MINER_PLAGIARISM_SCORE = 0.9  # want to make sure we're filtering out the bad actors
    PLAGIARISM_UPDATE_FREQUENCY_MS = 1000 * 60 * 60 # 1 hour
    PLAGIARISM_REVIEW_PERIOD_MS = 1000 * 60 * 60 * 24 * 14 # Time from plagiarism detection to elimination, 2 weeks
    PLAGIARISM_URL = "https://plagiarism.ultron.ts.taoshi.io/plagiarism" # Public domain for getting plagiarism scores

    BASE_DIR = base_directory = BASE_DIR

    METAGRAPH_UPDATE_REFRESH_TIME_VALIDATOR_MS = 60 * 1000  # 1 minute
    METAGRAPH_UPDATE_REFRESH_TIME_MINER_MS = 60 * 1000 * 15  # 15 minutes
    ELIMINATION_CHECK_INTERVAL_MS = 60 * 5 * 1000  # 5 minutes
    ELIMINATION_CACHE_REFRESH_INTERVAL_S = 5  # Elimination cache refresh interval in seconds
    ELIMINATION_FILE_DELETION_DELAY_MS = 7 * 24 * 60 * 60 * 1000  # 7 days

    # Entity Miners Configuration
    ENTITY_ELIMINATION_CHECK_INTERVAL = 300  # 5 minutes (in seconds) - for challenge period + elimination checks
    MAX_REGISTERED_ENTITIES = 5  # Maximum number of entities that can register
    ENTITY_MAX_SUBACCOUNTS = 10_000  # Default maximum subaccounts per entity (Phase 1)
    ENTITY_DATA_DIR = "validation/entities/"  # Entity data persistence directory
    FIXED_SUBACCOUNT_SIZE = 10000.0  # Fixed account size for subaccounts (USD) - placeholder
    SUBACCOUNT_COLLATERAL_AMOUNT = 1000.0  # Placeholder collateral amount per subaccount

    # Challenge Period Configuration
    SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD = 0.08  # 8% returns required to pass challenge period
    SUBACCOUNT_CHALLENGE_DRAWDOWN_THRESHOLD = 0.05  # 5% max drawdown allowed during challenge period

    # Subaccount promotion requirements
    SUBACCOUNT_FUNDED_MINIMUM_DAYS = 90  # Minimum days in FUNDED before promoting to ALPHA

    # Distributional statistics
    SOFTMAX_TEMPERATURE = 0.15

    # Qualifications to be a trusted validator sending checkpoints
    TOP_N_CHECKPOINTS = 10
    TOP_N_STAKE = 20
    STAKE_MIN = 1000.0
    AXON_NO_IP = "0.0.0.0"

    # Authorized mothership hotkey for state broadcasts
    # This is the ONLY validator authorized to broadcast CollateralRecord, AssetSelection, and SubaccountRegistration updates
    # TODO: Replace with actual mothership hotkey SS58 address
    MOTHERSHIP_HOTKEY = "5FeNwZ5oAqcJMitNqGx71vxGRWJhsdTqxFGVwPRfg8h2UZmo"
    MOTHERSHIP_HOTKEY_TESTNET = "5GTNzNkJiQWK4NpEErQohqZC8EzzeqrckgLgrQPwuvu8bHLN"
    # Require at least this many successful checkpoints before building golden
    MIN_CHECKPOINTS_RECEIVED = 5

    # Cap leverage across miner's entire portfolio
    PORTFOLIO_LEVERAGE_CAP = {
        TradePairCategory.CRYPTO: 5,
        TradePairCategory.FOREX: 20,
        TradePairCategory.INDICES: 10,
        TradePairCategory.EQUITIES: 2,
    }

    # Collateral limits
    MIN_COLLATERAL_BALANCE_THETA = 300  # Required minimum total collateral balance per miner in Theta. Approx $150k capital account size
    MAX_COLLATERAL_BALANCE_THETA = 1000  # Approx $500k capital account size
    MIN_COLLATERAL_BALANCE_TESTNET = 100
    MAX_COLLATERAL_BALANCE_TESTNET = 10000.0

    # Entity Miner Collateral
    ENTITY_REGISTRATION_FEE = 5000  # Theta required to register an entity
    ENTITY_COST_PER_THETA = 5000  # USD account size per theta of collateral for entity subaccounts
    MAX_SUBACCOUNT_ACCOUNT_SIZE = 100_000  # Maximum account size in USD for entity subaccounts

    # Account Size
    COST_PER_THETA = 500  # Account size USD value per theta of collateral
    MIN_COLLATERAL_VALUE = MIN_COLLATERAL_BALANCE_THETA * COST_PER_THETA   # Approx $150k
    MIN_CAPITAL = 5_000   # USD minimum capital account size
    DEFAULT_CAPITAL = 100_000  # conversion of 1x leverage to $100K in capital

    ANNUAL_INTEREST_RATE = 0.066  # 6.6%
    DAILY_INTEREST_RATE = ANNUAL_INTEREST_RATE / 365

    # 100% percent of collateral deposit is at risk of slashing based on drawdown
    DRAWDOWN_SLASH_PROPORTION = 1.0

    BLOCKED_TRADE_PAIR_IDS = {
        'SPX', 'DJI', 'NDX', 'VIX', 'FTSE', 'GDAXI',  # Indices
        'XAUUSD', 'XAGUSD',  # Commodities
        'AUDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'NZDJPY', 'GBPJPY', 'USDJPY',  # Forex JPY pairs
        'USDMXN'
    }

    # Trade pairs that are permanently unsupported (no price data available)
    # This constant is referenced by TradePair enum values after class definition
    UNSUPPORTED_TRADE_PAIRS = None  # Will be set after TradePair definition

    MAX_UNFILLED_LIMIT_ORDERS = 100
    LIMIT_ORDER_CHECK_REFRESH_MS = 10 * 1000 # 10 seconds
    LIMIT_ORDER_FILL_INTERVAL_MS = 30 * 1000 # 30 seconds

    LIMIT_ORDER_PRICE_BUFFER_TOLERANCE = 0.001 # +-0.1% tolerance
    LIMIT_ORDER_PRICE_BUFFER_MS = 30 * 1000
    MIN_UNIQUE_PRICES_FOR_LIMIT_FILL = 5

assert ValiConfig.CRYPTO_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.CRYPTO_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE
assert ValiConfig.FOREX_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.FOREX_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE
assert ValiConfig.INDICES_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.INDICES_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE
assert ValiConfig.EQUITIES_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.EQUITIES_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE

class TradePair(Enum):
    # crypto
    BTCUSD = ["BTCUSD", "BTC/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO, CryptoSubcategory.MAJORS]
    ETHUSD = ["ETHUSD", "ETH/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO, CryptoSubcategory.MAJORS]
    SOLUSD = ["SOLUSD", "SOL/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO, CryptoSubcategory.ALTS]
    XRPUSD = ["XRPUSD", "XRP/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
                TradePairCategory.CRYPTO, CryptoSubcategory.ALTS]
    DOGEUSD = ["DOGEUSD", "DOGE/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
                TradePairCategory.CRYPTO, CryptoSubcategory.ALTS]
    ADAUSD = ["ADAUSD", "ADA/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
               TradePairCategory.CRYPTO, CryptoSubcategory.ALTS]

    # TAO (data-only, not tradeable - used for emissions ledger calculations)
    TAOUSD = ["TAOUSD", "TAO/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO, CryptoSubcategory.ALTS]


    # forex
    AUDCAD = ["AUDCAD", "AUD/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G5]
    AUDCHF = ["AUDCHF", "AUD/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G5]
    AUDUSD = ["AUDUSD", "AUD/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G1]
    AUDJPY = ["AUDJPY", "AUD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G2]
    AUDNZD = ["AUDNZD", "AUD/NZD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G5]
    CADCHF = ["CADCHF", "CAD/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G5]
    CADJPY = ["CADJPY", "CAD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G2]
    CHFJPY = ["CHFJPY", "CHF/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G2]
    EURAUD = ["EURAUD", "EUR/AUD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G3]
    EURCAD = ["EURCAD", "EUR/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G3]
    EURUSD = ["EURUSD", "EUR/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G1]
    EURCHF = ["EURCHF", "EUR/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G3]
    EURGBP = ["EURGBP", "EUR/GBP", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G3]
    EURJPY = ["EURJPY", "EUR/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G2]
    EURNZD = ["EURNZD", "EUR/NZD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G3]
    NZDCAD = ["NZDCAD", "NZD/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G5]
    NZDCHF = ["NZDCHF", "NZD/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G5]
    NZDJPY = ["NZDJPY", "NZD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G2]
    NZDUSD = ["NZDUSD", "NZD/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G1]
    GBPAUD = ["GBPAUD", "GBP/AUD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
                TradePairCategory.FOREX, ForexSubcategory.G4]
    GBPCAD = ["GBPCAD", "GBP/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
                TradePairCategory.FOREX, ForexSubcategory.G4]
    GBPCHF = ["GBPCHF", "GBP/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G4]
    GBPJPY = ["GBPJPY", "GBP/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G2]
    GBPNZD = ["GBPNZD", "GBP/NZD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G4]
    GBPUSD = ["GBPUSD", "GBP/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G1]
    USDCAD = ["USDCAD", "USD/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G1]
    USDCHF = ["USDCHF", "USD/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G1]
    USDJPY = ["USDJPY", "USD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G2]
    USDMXN = ["USDMXN", "USD/MXN", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX, ForexSubcategory.G5]


    # "Commodities" (Bundle with Forex for now) (temporariliy paused for trading)
    XAUUSD = ["XAUUSD", "XAU/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE, TradePairCategory.FOREX]
    XAGUSD = ["XAGUSD", "XAG/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE, TradePairCategory.FOREX]

    # Equities - Stocks
    # Technology (10)
    NVDA = ["NVDA", "NVDA", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    MSFT = ["MSFT", "MSFT", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    AAPL = ["AAPL", "AAPL", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    AVGO = ["AVGO", "AVGO", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    TSM = ["TSM", "TSM", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    ORCL = ["ORCL", "ORCL", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    AMD = ["AMD", "AMD", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    MU = ["MU", "MU", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    CRM = ["CRM", "CRM", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    UBER = ["UBER", "UBER", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    # Financial Services (5)
    BRK_B = ["BRK_B", "BRK.B", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    JPM = ["JPM", "JPM", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    V = ["V", "V", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    MA = ["MA", "MA", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    BAC = ["BAC", "BAC", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    # Consumer Discretionary (5)
    AMZN = ["AMZN", "AMZN", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    TSLA = ["TSLA", "TSLA", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    HD = ["HD", "HD", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    BABA = ["BABA", "BABA", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    SBUX = ["SBUX", "SBUX", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    # Communication Services (5)
    GOOGL = ["GOOGL", "GOOGL", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    META = ["META", "META", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    NFLX = ["NFLX", "NFLX", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    APP = ["APP", "APP", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    T = ["T", "T", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]

    # Equities - Sector ETFs (22)
    XLK = ["XLK", "XLK", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VGT = ["VGT", "VGT", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLF = ["XLF", "XLF", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VFH = ["VFH", "VFH", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLY = ["XLY", "XLY", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VCR = ["VCR", "VCR", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLC = ["XLC", "XLC", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VOX = ["VOX", "VOX", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLV = ["XLV", "XLV", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VHT = ["VHT", "VHT", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLI = ["XLI", "XLI", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VIS = ["VIS", "VIS", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLP = ["XLP", "XLP", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VDC = ["VDC", "VDC", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLE = ["XLE", "XLE", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VDE = ["VDE", "VDE", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLB = ["XLB", "XLB", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VAW = ["VAW", "VAW", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLU = ["XLU", "XLU", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VPU = ["VPU", "VPU", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    XLRE = ["XLRE", "XLRE", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    VNQ = ["VNQ", "VNQ", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]


    # indices (no longer allowed for trading as we moved to equities tickers instead)
    SPX = ["SPX", "SPX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    DJI = ["DJI", "DJI", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    NDX = ["NDX", "NDX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    VIX = ["VIX", "VIX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    FTSE = ["FTSE", "FTSE", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
            TradePairCategory.INDICES]
    GDAXI = ["GDAXI", "GDAXI", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
             TradePairCategory.INDICES]

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
        if len(self.value) > 6:
            return self.value[6]
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
    def is_blocked(self) -> bool:
        """Check if this trade pair is blocked from trading"""
        return self.trade_pair_id in ValiConfig.BLOCKED_TRADE_PAIR_IDS

    @property
    def lot_size(self):
        trade_pair_lot_size = {TradePairCategory.CRYPTO: 1,
                               TradePairCategory.FOREX: 100_000,
                               TradePairCategory.INDICES: 1,
                               TradePairCategory.EQUITIES: 1}
        return trade_pair_lot_size[self.trade_pair_category]

    @property
    def base(self):
        return self.trade_pair.split("/")[0]

    @property
    def quote(self):
        if self.is_forex:
            return self.trade_pair.split("/")[1]
        else:
            return "USD"

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
            TradePair: The corresponding TradePair object.

        Raises:
            ValueError: If no matching trade pair is found.
        """
        if trade_pair_id in TRADE_PAIR_ID_TO_TRADE_PAIR:
            return TRADE_PAIR_ID_TO_TRADE_PAIR[trade_pair_id]
        else:
            return None

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

    def __dict__(self):
        return self.__json__()

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

# Set UNSUPPORTED_TRADE_PAIRS now that TradePair enum is defined
# These are trade pairs that have no price data available (not just temporarily halted)
ValiConfig.UNSUPPORTED_TRADE_PAIRS = (TradePair.SPX, TradePair.DJI, TradePair.NDX, TradePair.VIX,
                                      TradePair.FTSE, TradePair.GDAXI, TradePair.TAOUSD)
