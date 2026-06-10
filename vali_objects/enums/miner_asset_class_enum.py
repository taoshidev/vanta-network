from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vali_objects.vali_config import TradePair


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

    @property
    def is_multi_class(self) -> bool:
        """
        True if this asset class represents a multi-class subaccount (one that may hold
        positions across more than one TradePairCategory). Multi-class subaccounts are
        subject to per-class portfolio sub-caps plus a tighter overall cap; single-class
        subaccounts use a single per-class cap.
        """
        return self == MinerAssetClass.HL_ALL

    @staticmethod
    def is_valid(asset_class: str) -> bool:
        """True if `asset_class` (case-insensitive) is a valid MinerAssetClass value."""
        if not isinstance(asset_class, str):
            return False
        return asset_class.lower() in {c.value for c in MinerAssetClass}

    def allows_trade_pair(self, trade_pair: "TradePair") -> bool:
        """
        Check if `trade_pair` is allowed for this miner asset class.

        - HL_ALL allows Hyperliquid pairs plus forex (except XAUUSD/XAGUSD)
        - COMMODITIES requires Hyperliquid source and commodity category
        - Other classes require Vanta source and matching category
        """
        from vali_objects.vali_config import TradePair, TradePairCategory, TradePairSource

        category = trade_pair.trade_pair_category
        src = trade_pair.src

        if self == MinerAssetClass.HL_ALL:
            excluded_forex = {TradePair.XAUUSD, TradePair.XAGUSD}
            is_supported_forex = category == TradePairCategory.FOREX and trade_pair not in excluded_forex
            return src == TradePairSource.HYPERLIQUID or is_supported_forex

        if self == MinerAssetClass.COMMODITIES:
            return src == TradePairSource.HYPERLIQUID and category == TradePairCategory.COMMODITIES

        return src == TradePairSource.VANTA and self.value == category.value


