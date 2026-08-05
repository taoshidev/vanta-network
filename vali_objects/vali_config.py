# developer: Taoshi
from datetime import datetime, timezone
import os
import math
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


from vali_objects.enums.miner_asset_class_enum import MinerAssetClass  # noqa: E402,F401

# Re-export TradePair classes for backwards compatibility.
# Per-trade-pair constants (leverage, BLOCKED/FLAT_ONLY id sets, lookup dicts) live in
# vali_objects.trade_pair and should be imported from there directly.
from vali_objects.trade_pair import (  # noqa: E402,F401
    TradePair,
    TradePairCategory,
    TradePairSource,
    InstrumentType,
)



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
    VANTA_CLI_MINIMUM_VERSION = "2.2.1"

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

    RPC_HL_FUNDING_PORT = 50025
    RPC_HL_FUNDING_SERVICE_NAME = "HLFundingRateServer"

    RPC_ENTITY_COLLATERAL_PORT = 50026
    RPC_ENTITY_COLLATERAL_SERVICE_NAME = "EntityCollateralServer"

    RPC_MARKET_ORDER_PORT = 50027
    RPC_MARKET_ORDER_SERVICE_NAME = "MarketOrderServer"

    # Entity collateral cache refresh interval (seconds)
    ENTITY_COLLATERAL_CACHE_REFRESH_S = 30 * 60

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
    TARGET_LEDGER_WINDOW_DAYS = 180
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
    MDD_CHECK_REFRESH_TIME_MS = 30 * 1000  # 30 seconds
    CHALLENGE_PERIOD_REFRESH_TIME_MS = MDD_CHECK_REFRESH_TIME_MS
    PRICE_SOURCE_COMPACTING_SLEEP_INTERVAL_SECONDS = 60 * 60 * 12 # 12 hours

    # HL dynamic universe — HS position leverage mapping
    HL_HIGH_TIER_THRESHOLD = 50         # HL max lev at which HS high tier applies
    HS_HIGH_TIER_MAX_LEVERAGE = 5.0     # intended for forex/spx-tier (HL max lev 50x) pairs; dead code since forex pairs are excluded for now
    HS_PORTFOLIO_MAX_LEVERAGE = 4.0     # HS portfolio-level leverage cap (funded accounts)

    # Minimum position size
    FOREX_MIN_POSITION_SIZE_LOTS = 0.01            # micro lot — subaccounts above FOREX_SMALL_ACCOUNT_THRESHOLD
    FOREX_MIN_POSITION_SIZE_LOTS_NANO = 0.001      # nano lot — deprecated; no account tier currently uses this
    FOREX_MIN_POSITION_SIZE_LOTS_SUB_NANO = 0.0001 # sub-nano lot — subaccounts at or below FOREX_SMALL_ACCOUNT_THRESHOLD
    FOREX_SMALL_ACCOUNT_THRESHOLD = 10_000.0       # USD; subaccounts at or below this use sub-nano lot minimum
    CRYPTO_MIN_POSITION_SIZE_USD = 10.0  # $10 USD
    EQUITIES_MIN_POSITION_SIZE_SHARES = 0.01 # 0.01 shares
    DEFAULT_MIN_POSITION_SIZE_USD = 10.0

    # Minimum order size in quantity (different from minimum position size ex. crypto)
    CRYPTO_MIN_ORDER_SIZE = 0.00001
    COMMODITIES_MIN_ORDER_SIZE = 0.00001
    EQUITIES_MIN_ORDER_SIZE = 0.00001
    FOREX_MIN_ORDER_SIZE = 0.01
    FOREX_MIN_ORDER_SIZE_SUB_NANO = 0.0001

    MAX_DAILY_DRAWDOWN = 0.95  # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9  # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_TOTAL_DRAWDOWN_V2 = 0.95
    MAX_ORDERS_PER_POSITION = 100
    ORDER_COOLDOWN_MS = 5000  # 5 seconds
    ORDER_MIN_LEVERAGE = 0.00001
    ORDER_MAX_LEVERAGE = 500

    # Controls how much history to store for price data which is used in retroactive updates
    RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS = 300000 # 5 minutes

    # Risk Profiling
    RISK_PROFILING_STEPS_MIN_LEVERAGE = 0.01  # min of category MIN_LEVERAGE values in vali_objects/trade_pair.py
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
    CHALLENGE_PERIOD_MINIMUM_MS = CHALLENGE_PERIOD_MINIMUM_DAYS * DAILY_MS
    CHALLENGE_PERIOD_MAXIMUM_DAYS = 90
    CHALLENGE_PERIOD_MAXIMUM_MS = CHALLENGE_PERIOD_MAXIMUM_DAYS * DAILY_MS
    CHALLENGE_PERIOD_PERCENTILE_THRESHOLD = 0.75 # miners must pass 75th percentile to enter the main competition

    PROBATION_MAXIMUM_DAYS = 90
    PROBATION_MAXIMUM_MS = PROBATION_MAXIMUM_DAYS * DAILY_MS

    IDLE_MINER_MAXIMUM_DAYS = 60
    IDLE_MINER_MAXIMUM_MS = IDLE_MINER_MAXIMUM_DAYS * DAILY_MS

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
    ELIMINATION_CHECK_INTERVAL_MS = 60 * 2 * 1000  # 2 minutes
    ELIMINATION_CACHE_REFRESH_INTERVAL_S = 5  # Elimination cache refresh interval in seconds
    ELIMINATION_FILE_DELETION_DELAY_MS = 30 * 24 * 60 * 60 * 1000  # 30 days

    # Entity Miners Configuration
    ENTITY_ELIMINATION_CHECK_INTERVAL = 300  # 5 minutes (in seconds) - for challenge period + elimination checks
    MAX_REGISTERED_ENTITIES = 10  # Maximum number of entities that can register
    ENTITY_MAX_SUBACCOUNTS = 10_000  # Default maximum subaccounts per entity (Phase 1)
    ENTITY_DATA_DIR = "validation/entities/"  # Entity data persistence directory
    FIXED_SUBACCOUNT_SIZE = 10000.0  # Fixed account size for subaccounts (USD) - placeholder
    SUBACCOUNT_COLLATERAL_AMOUNT = 1000.0  # Placeholder collateral amount per subaccount

    # Challenge Period Configuration
    SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD_DEFAULT = 0.1  # Default fallback returns threshold
    SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD = {
        MinerAssetClass.CRYPTO: 0.1,      # 10% returns required to pass crypto evaluation
        MinerAssetClass.FOREX: 0.08,      # 8% returns required to pass forex evaluation
        MinerAssetClass.EQUITIES: 0.1,    # 10% returns required to pass equities evaluation
        MinerAssetClass.HL_ALL: 0.1,      # 10% returns required to pass hl all markets evaluation
        MinerAssetClass.ALL_MARKETS: 0.1, # 10% returns required to pass all markets evaluation
        MinerAssetClass.COMMODITIES: 0.1, # 10% returns required to pass commodities evaluation
    }
    CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD = 0.05    # Rule 1: 5% intraday drop from day-open equity eliminates
    CHALLENGE_EOD_DRAWDOWN_THRESHOLD = 0.05  # Rule 2: 5% drop from highest-ever EOD equity eliminates
    FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V0 = 0.10 # V0 applies to subaccounts registered before Sun Mar 15, 2026
    FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V1 = 0.08 # V1 applies to subaccounts registered before Wed May 27, 2026
    FUNDED_INTRADAY_DRAWDOWN_THRESHOLD = 0.05
    FUNDED_EOD_DRAWDOWN_THRESHOLD_V0 = 0.10  # V0 applies to subaccounts registered before Sun Mar 15, 2026
    FUNDED_EOD_DRAWDOWN_THRESHOLD = 0.08  # Rule 2: 8% drop from highest-ever EOD equity (trailing) eliminates

    # Registration cutoffs for versioned SUBACCOUNT_FUNDED thresholds (ms)
    FUNDED_V0_CUTOFF_MS = 1773532799000  # Mar 14, 2026 23:59:59 UTC
    FUNDED_V1_CUTOFF_MS = 1779840000000  # May 27, 2026 00:00:00 UTC

    # Entity subaccount (standard account) static drawdown rules
    SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD = 0.05  # Rule 1: balance (excl. unrealized PnL) more than 5% below starting balance eliminates
    SUBACCOUNT_STATIC_EOD_DRAWDOWN_THRESHOLD = 0.05  # Rule 2: equity (incl. unrealized PnL) more than 5% below starting balance at 00:00 UTC eliminates

    # Subaccount promotion requirements
    SUBACCOUNT_FUNDED_MINIMUM_DAYS = 90  # Minimum days in FUNDED before promoting to ALPHA

    # Minimum tier required to get checkpoint file
    CHECKPOINT_TIER = 100

    # Minimum tier required for subaccount dashboard subscriptions
    SUBACCOUNT_SUBSCRIPTION_TIER = 200

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

    # Account size thresholds for leverage tier progression (non-challenge entity subaccounts)
    LEVERAGE_TIER3_MIN_ACCOUNT_SIZE = 200_000    # $200K: Tier 2 → Tier 3
    LEVERAGE_TIER4_MIN_ACCOUNT_SIZE = 1_000_000  # $1M:   Tier 3 → Tier 4

    # Cap leverage across an individual miner's entire portfolio, per pair.
    # Keyed on (asset class, instrument type).
    PORTFOLIO_LEVERAGE_CAP = {
        (TradePairCategory.CRYPTO,      InstrumentType.SPOT): 5,
        (TradePairCategory.CRYPTO,      InstrumentType.PERP): 5,
        (TradePairCategory.FOREX,       InstrumentType.SPOT): 20,
        (TradePairCategory.FOREX,       InstrumentType.PERP): 20,
        (TradePairCategory.EQUITIES,    InstrumentType.SPOT): 2,    # Reg T overnight
        (TradePairCategory.EQUITIES,    InstrumentType.PERP): 5,
        (TradePairCategory.INDICES,     InstrumentType.SPOT): 10,
        (TradePairCategory.INDICES,     InstrumentType.PERP): 5,
        (TradePairCategory.COMMODITIES, InstrumentType.SPOT): 5,
        (TradePairCategory.COMMODITIES, InstrumentType.PERP): 5,
    }

    # Per-tier portfolio leverage caps. Split into two dicts because the underlying lookup is
    # semantically two different things:
    #   *_BY_PAIR        : per-pair multiplier used when validating an incoming order in
    #                      market_order_manager. Keyed on (asset class, instrument type).
    #   *_BY_ASSET_CLASS : account-wide multiplier from the subaccount's own asset_class field
    #                      (which can be HL_ALL). Keyed by single MinerAssetClass.
    # XAUUSD/XAGUSD positions land in the FOREX subaccount asset_class bucket.
    # Equity portfolio cap stays 2x from Tier 3 onward in the SPOT column (Reg T overnight).
    TIER_PORTFOLIO_LEVERAGE_BY_PAIR = {
        1: {
            (TradePairCategory.CRYPTO,      InstrumentType.SPOT): 2.0,
            (TradePairCategory.CRYPTO,      InstrumentType.PERP): 2.0,
            (TradePairCategory.FOREX,       InstrumentType.SPOT): 5.0,
            (TradePairCategory.FOREX,       InstrumentType.PERP): 5.0,
            (TradePairCategory.EQUITIES,    InstrumentType.SPOT): 1.0,
            (TradePairCategory.EQUITIES,    InstrumentType.PERP): 2.0,
            (TradePairCategory.INDICES,     InstrumentType.SPOT): 5.0,
            (TradePairCategory.INDICES,     InstrumentType.PERP): 2.0,
            (TradePairCategory.COMMODITIES, InstrumentType.SPOT): 2.0,
            (TradePairCategory.COMMODITIES, InstrumentType.PERP): 2.0,
        },
        2: {
            (TradePairCategory.CRYPTO,      InstrumentType.SPOT): 2.0,
            (TradePairCategory.CRYPTO,      InstrumentType.PERP): 2.0,
            (TradePairCategory.FOREX,       InstrumentType.SPOT): 10.0,
            (TradePairCategory.FOREX,       InstrumentType.PERP): 10.0,
            (TradePairCategory.EQUITIES,    InstrumentType.SPOT): 1.5,
            (TradePairCategory.EQUITIES,    InstrumentType.PERP): 2.0,
            (TradePairCategory.INDICES,     InstrumentType.SPOT): 10.0,
            (TradePairCategory.INDICES,     InstrumentType.PERP): 2.0,
            (TradePairCategory.COMMODITIES, InstrumentType.SPOT): 2.0,
            (TradePairCategory.COMMODITIES, InstrumentType.PERP): 2.0,
        },
        3: {
            (TradePairCategory.CRYPTO,      InstrumentType.SPOT): 3.0,
            (TradePairCategory.CRYPTO,      InstrumentType.PERP): 3.0,
            (TradePairCategory.FOREX,       InstrumentType.SPOT): 15.0,
            (TradePairCategory.FOREX,       InstrumentType.PERP): 15.0,
            (TradePairCategory.EQUITIES,    InstrumentType.SPOT): 2.0,
            (TradePairCategory.EQUITIES,    InstrumentType.PERP): 3.0,
            (TradePairCategory.INDICES,     InstrumentType.SPOT): 15.0,
            (TradePairCategory.INDICES,     InstrumentType.PERP): 3.0,
            (TradePairCategory.COMMODITIES, InstrumentType.SPOT): 3.0,
            (TradePairCategory.COMMODITIES, InstrumentType.PERP): 3.0,
        },
        4: {
            (TradePairCategory.CRYPTO,      InstrumentType.SPOT): 4.0,
            (TradePairCategory.CRYPTO,      InstrumentType.PERP): 4.0,
            (TradePairCategory.FOREX,       InstrumentType.SPOT): 20.0,
            (TradePairCategory.FOREX,       InstrumentType.PERP): 20.0,
            (TradePairCategory.EQUITIES,    InstrumentType.SPOT): 2.0,
            (TradePairCategory.EQUITIES,    InstrumentType.PERP): 4.0,
            (TradePairCategory.INDICES,     InstrumentType.SPOT): 20.0,
            (TradePairCategory.INDICES,     InstrumentType.PERP): 4.0,
            (TradePairCategory.COMMODITIES, InstrumentType.SPOT): 4.0,
            (TradePairCategory.COMMODITIES, InstrumentType.PERP): 4.0,
        },
    }

    # Single-class per-category sub-caps. Multi-class subaccounts (HL_ALL, ALL_MARKETS) reuse
    # these per-class entries for sub-cap enforcement, and pull their overall cross-class cap
    # from the HL_ALL/ALL_MARKETS entries in TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS below.
    TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY = {
        1: {TradePairCategory.CRYPTO: 2.0, TradePairCategory.FOREX: 5.0,  TradePairCategory.EQUITIES: 1.0, TradePairCategory.INDICES: 3.0,  TradePairCategory.COMMODITIES: 2.0},
        2: {TradePairCategory.CRYPTO: 2.0, TradePairCategory.FOREX: 10.0, TradePairCategory.EQUITIES: 1.5, TradePairCategory.INDICES: 6.0,  TradePairCategory.COMMODITIES: 2.0},
        3: {TradePairCategory.CRYPTO: 3.0, TradePairCategory.FOREX: 15.0, TradePairCategory.EQUITIES: 2.0, TradePairCategory.INDICES: 8.0,  TradePairCategory.COMMODITIES: 3.0},
        4: {TradePairCategory.CRYPTO: 4.0, TradePairCategory.FOREX: 20.0, TradePairCategory.EQUITIES: 2.0, TradePairCategory.INDICES: 10.0, TradePairCategory.COMMODITIES: 4.0},
    }
    TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS = {
        1: {MinerAssetClass.CRYPTO: 2.0, MinerAssetClass.FOREX: 5.0,  MinerAssetClass.EQUITIES: 1.0, MinerAssetClass.COMMODITIES: 2.0, MinerAssetClass.HL_ALL: 4.0,  MinerAssetClass.ALL_MARKETS: 6.0},
        2: {MinerAssetClass.CRYPTO: 2.0, MinerAssetClass.FOREX: 10.0, MinerAssetClass.EQUITIES: 1.5, MinerAssetClass.COMMODITIES: 2.0, MinerAssetClass.HL_ALL: 7.0, MinerAssetClass.ALL_MARKETS: 12.0},
        3: {MinerAssetClass.CRYPTO: 3.0, MinerAssetClass.FOREX: 15.0, MinerAssetClass.EQUITIES: 2.0, MinerAssetClass.COMMODITIES: 3.0, MinerAssetClass.HL_ALL: 10.0, MinerAssetClass.ALL_MARKETS: 18.0},
        4: {MinerAssetClass.CRYPTO: 4.0, MinerAssetClass.FOREX: 20.0, MinerAssetClass.EQUITIES: 2.0, MinerAssetClass.COMMODITIES: 4.0, MinerAssetClass.HL_ALL: 12.0, MinerAssetClass.ALL_MARKETS: 24.0},
    }

    # Collateral limits
    MIN_COLLATERAL_BALANCE_THETA = 300  # Required minimum total collateral balance per miner in Theta. Approx $150k capital account size
    MAX_COLLATERAL_BALANCE_THETA = 1000  # Approx $500k capital account size
    MIN_COLLATERAL_BALANCE_TESTNET = 100
    MAX_COLLATERAL_BALANCE_TESTNET = 10000.0

    # Entity Miner Collateral
    ENTITY_REGISTRATION_FEE = 1000  # Theta required to register an entity
    ENTITY_COST_PER_THETA = 5000  # USD account size per theta of collateral for entity subaccounts
    ENTITY_COST_PER_THETA_LOW = 2500  # CPT value used for smaller account sizes <=10k
    ENTITY_COST_PER_THETA_LOW_THRESHOLD = 10_000  # Account sizes at or below this use ENTITY_COST_PER_THETA_LOW
    MAX_SUBACCOUNT_ACCOUNT_SIZE = 100_000  # Maximum account size in USD for entity subaccounts

    # Entity margin collateral requirement (funded subaccounts only):
    #   required_theta = sum(max_slash_usd - cumulative_slashed_usd) / CPT_RISK
    #   for each funded subaccount with open positions (or placing this order)
    # max_slash_usd = account_size * SUBACCOUNT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD
    ENTITY_COLLATERAL_CPT_RISK = 35  # USD of remaining loss capacity per theta ($35 of capacity = 1 theta)

    # Hyperliquid tracking configuration
    HL_USE_TESTNET = False  # Set to True to use Hyperliquid testnet endpoints
    HL_MAINNET_WS = "wss://api.hyperliquid.xyz/ws"
    HL_MAINNET_INFO = "https://api.hyperliquid.xyz/info"
    HL_MAINNET_HOST = "api.hyperliquid.xyz"

    HL_TESTNET_WS = "wss://api.hyperliquid-testnet.xyz/ws"
    HL_TESTNET_INFO = "https://api.hyperliquid-testnet.xyz/info"
    HL_TESTNET_HOST = "api.hyperliquid-testnet.xyz"

    @classmethod
    def hl_ws_url(cls) -> str:
        return cls.HL_TESTNET_WS if cls.HL_USE_TESTNET else cls.HL_MAINNET_WS

    @classmethod
    def hl_info_url(cls) -> str:
        return cls.HL_TESTNET_INFO if cls.HL_USE_TESTNET else cls.HL_MAINNET_INFO

    @classmethod
    def hl_host(cls) -> str:
        return cls.HL_TESTNET_HOST if cls.HL_USE_TESTNET else cls.HL_MAINNET_HOST

    HL_MAX_TRACKED_ADDRESSES_PER_IP = 10  # HL WebSocket limit: 10 unique users per IP
    HL_MAX_TRACKED_ADDRESSES = HL_MAX_TRACKED_ADDRESSES_PER_IP  # backward compat alias
    HL_WS_HEARTBEAT_INTERVAL_S = 30.0
    HL_WS_RECONNECT_BACKOFF_MAX_S = 30.0
    HL_PROXY_SECRET_KEY = "hl_proxy_url"  # key in secrets.json for base proxy URL (without port)
    HL_PROXY_PORTS_SECRET_KEY = "hl_proxy_ports"  # key in secrets.json for port list/range
    HL_MAX_PROXY_SHARDS = 100  # safety cap on proxy connections (500 addresses max)
    HL_SHARD_MAX_CONSECUTIVE_FAILURES = 5  # failures before marking a proxy IP as unhealthy
    HL_PORT_REST_FAILURE_THRESHOLD = 3
    HL_PORT_HEALTH_PROBE_INTERVAL_S = 30.0
    HL_PORT_HEALTH_MAX_COOLDOWN_S = 600.0
    HL_ADDRESS_REGEX = r"^0x[a-fA-F0-9]{40}$"
    HL_BACKUP_POLL_INTERVAL_S = 10.0
    HL_BACKUP_POLL_RATE_BUDGET = 60
    HL_BACKUP_POLL_LOOKBACK_MS = 60 * 60 * 1000 # TODO: change to 2 min
    HL_BACKUP_RESTART_LOOKBACK_MS = 60 * 60 * 1000

    # L2 orderbook precision: nSigFigs controls price aggregation granularity.
    # HL returns max 20 levels per side regardless of nSigFigs.
    # None (full precision, native ticks) is finest; 2 sig figs is coarsest/deepest
    # coverage but loses granular price distribution. We subscribe to every resolution
    # in the cascade on its own WS connection and walk them finest-to-coarsest, only
    # extending into the next coarser book once the finer one is exhausted.
    HL_L2_SIG_FIGS_CASCADE = [None, 5, 4, 3, 2]

    # Slippage audit logging: log the L2 levels walked whenever simulated Hyperliquid
    # slippage exceeds this fraction, so anomalously high slippage can be
    # reconstructed/verified after the fact.
    HL_SLIPPAGE_AUDIT_LOG_THRESHOLD = 0.008

    # HL Funding Rate Service
    HL_FUNDING_DAEMON_INTERVAL_S = 300
    HL_FUNDING_BACKFILL_HOURS = 4

    # Account Size
    COST_PER_THETA = 500  # Account size USD value per theta of collateral
    MIN_COLLATERAL_VALUE = MIN_COLLATERAL_BALANCE_THETA * COST_PER_THETA   # Approx $150k
    MIN_CAPITAL = 5_000   # USD minimum capital account size
    DEFAULT_CAPITAL = 100_000  # conversion of 1x leverage to $100K in capital

    # 100% percent of collateral deposit is at risk of slashing based on drawdown
    DRAWDOWN_SLASH_PROPORTION = 1.0

    MAX_UNFILLED_LIMIT_ORDERS = 100
    LIMIT_ORDER_FILL_INTERVAL_MS = 10 * 1000 # 10 seconds
    LIMIT_ORDER_PRICE_BUFFER_MS = 30 * 1000

