from enum import Enum
from vali_objects.trade_pair import TradePair, TradePairCategory, TradePairSource

class MinerAssetClass(str, Enum):
    """
    Asset class a miner's subaccount is registered under.

    Distinct from TradePairCategory: this enum describes the bucket a *miner*
    selects, while TradePairCategory describes the intrinsic category of a
    *trade pair*. Indices have no MinerAssetClass — index pairs are blocked
    and HL index perps are accessed via HL_ALL subaccounts.
    """
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITIES = "equities"
    COMMODITIES = "commodities"
    HL_ALL = "hl_all"
    ALL_MARKETS = "all_markets"

    @staticmethod
    def is_valid(asset_class: str) -> bool:
        """True if `asset_class` (case-insensitive) is a valid MinerAssetClass value."""
        if not isinstance(asset_class, str):
            return False
        return asset_class.lower() in {c.value for c in MinerAssetClass}

    def can_trade(self, trade_pair: TradePair) -> bool:
        """
        Check if `trade_pair` is allowed for this miner asset class.

        - HL_ALL allows Hyperliquid pairs plus forex (except XAUUSD/XAGUSD)
        - COMMODITIES requires Hyperliquid source and commodity category
        - Other classes require Vanta source and matching category
        """

        category = trade_pair.trade_pair_category
        src = trade_pair.src

        if trade_pair.is_blocked:
            return False

        if self == MinerAssetClass.HL_ALL:
            return src == TradePairSource.HYPERLIQUID

        if self == MinerAssetClass.ALL_MARKETS:
            excluded_forex = {TradePair.XAUUSD, TradePair.XAGUSD}
            is_supported_forex = category == TradePairCategory.FOREX and trade_pair not in excluded_forex
            is_vanta_equities = src == TradePairSource.VANTA and category == TradePairCategory.EQUITIES
            return src == TradePairSource.HYPERLIQUID or is_supported_forex or is_vanta_equities

        if self == MinerAssetClass.COMMODITIES:
            return src == TradePairSource.HYPERLIQUID and category == TradePairCategory.COMMODITIES

        if self == MinerAssetClass.CRYPTO:
            return src == TradePairSource.HYPERLIQUID  and category == TradePairCategory.CRYPTO

        return src == TradePairSource.VANTA and self.value == category.value
