import copy
import math

import bittensor as bt

from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint
from vali_objects.vali_config import ValiConfig, TradePair, TradePairCategory



class AssetSegmentation:
    def __init__(self, miner_ledgers: dict[str, dict[str, PerfLedger]]):
        self.overall_ledgers = miner_ledgers
        self.asset_breakdown: dict[TradePairCategory, dict] = ValiConfig.ASSET_CLASS_BREAKDOWN
        self.asset_classes = AssetSegmentation.distill_asset_classes(self.asset_breakdown)

    @staticmethod
    def distill_asset_classes(
            asset_breakdown: dict[TradePairCategory, dict]
    ) -> set[TradePairCategory]:
        """
        Distills the asset classes from the asset breakdown.
        Returns a set of TradePairCategory.
        """
        return set(asset_breakdown.keys())

    def segmentation(self, asset_class: TradePairCategory) -> dict[str, PerfLedger]:
        """
        Segments the overall ledgers into asset classes and aggregates them.
        Returns a dictionary where keys are asset classes and values are aggregated PerfLedgers.
        """
        if asset_class not in self.asset_classes:
            raise ValueError(f"Asset class {asset_class} is not recognized.")

        # Initialize segmented ledgers for the specified asset class
        subset = self.ledger_subset(asset_class)

        total_miner_ledgers = {}
        for hotkey, full_ledger in subset.items():
            portfolio_ledger = self.overall_ledgers.get(hotkey, {}).get("portfolio", PerfLedger())
            total_miner_ledgers[hotkey] = AssetSegmentation.aggregate_miner_subledgers(
                portfolio_ledger,
                full_ledger,
            )

        return total_miner_ledgers

    def ledger_subset(self, asset_class: TradePairCategory) -> dict[str, dict[str, PerfLedger]]:
        """
        Only returns the subset of ledgers that match the specified asset class.
        """
        if asset_class not in self.asset_classes:
            raise ValueError(f"Asset class {asset_class} is not recognized.")

        subset_ledger = {}
        for hotkey, full_ledger in self.overall_ledgers.items():
            if full_ledger is None:
                continue
            miner_subset_ledger = {}
            for asset_name, ledger in full_ledger.items():
                if asset_name == "portfolio":
                    continue

                trade_pair = TradePair.from_trade_pair_id(asset_name)
                if trade_pair.trade_pair_category == asset_class:
                    miner_subset_ledger[asset_name] = ledger

            subset_ledger[hotkey] = miner_subset_ledger

        return subset_ledger

    @staticmethod
    def aggregate_miner_subledgers(
            default_ledger: PerfLedger,
            sub_ledgers: dict[str, PerfLedger]
    ) -> PerfLedger:
        """
        Aggregates the sub-ledgers for a specific miner into a single PerfLedger.
        """
        if len(sub_ledgers) == 0:
            return PerfLedger()

        default_ledger_copy = copy.deepcopy(default_ledger)
        default_ledger_copy.cps: list[PerfCheckpoint] = []

        aggregated_dict_ledger = {}
        for _, ledger in sub_ledgers.items():
            ledger_checkpoints = ledger.cps
            for checkpoint in ledger_checkpoints:
                if checkpoint.last_update_ms not in aggregated_dict_ledger:
                    aggregated_dict_ledger[checkpoint.last_update_ms] = copy.deepcopy(checkpoint)
                else:
                    existing_checkpoint = aggregated_dict_ledger.get(checkpoint.last_update_ms)

                    # Aggregate the values of the existing checkpoint with the new one
                    existing_checkpoint.n_updates += checkpoint.n_updates
                    existing_checkpoint.gain += checkpoint.gain
                    existing_checkpoint.loss += checkpoint.loss
                    existing_checkpoint.spread_fee_loss += checkpoint.spread_fee_loss
                    existing_checkpoint.carry_fee_loss += checkpoint.carry_fee_loss
                    existing_checkpoint.realized_pnl += checkpoint.realized_pnl
                    existing_checkpoint.unrealized_pnl += checkpoint.unrealized_pnl

                    aggregated_dict_ledger[checkpoint.last_update_ms] = existing_checkpoint

        default_ledger_copy.cps = sorted(list(aggregated_dict_ledger.values()), key=lambda x: x.last_update_ms)
        return default_ledger_copy

    @staticmethod
    def segment_competitiveness(incentive_distribution: list[float]) -> float | None:
        """
        Indicates the relative level of competitiveness for an asset class based on the incentive distribution.
        """
        # Placeholder logic for competitiveness segmentation
        vals = incentive_distribution
        if vals is None:
            raise ValueError("Vals must not be None.")

        n = len(vals)
        if n == 0:
            return None
        if any(v < 0 for v in vals):
            raise ValueError("Gini coefficient is undefined for negative values")

        vals.sort()
        cumulative = 0.0
        cumulative_weighted = 0.0
        for i, v in enumerate(vals, 1):
            cumulative += v
            cumulative_weighted += i * v

        total = cumulative
        if n == 0 or total == 0:
            return None
        g = (2 * cumulative_weighted) / (n * total) - (n + 1) / n
        return g if not math.isnan(g) else None

    @staticmethod
    def asset_competitiveness_dictionary(
            asset_incentive_distributions: dict[str, dict[str, float]]
    ) -> dict[str, float | None]:
        """
        Returns a dictionary with asset classes as keys and their competitiveness as values.
        """
        competitiveness_dict = {}
        for asset_class, distribution in asset_incentive_distributions.items():
            if not distribution:
                bt.logging.warning(f"Distribution for {asset_class} isn't defined.")
                competitiveness_dict[asset_class] = None
            else:
                incentive_distribution = [value for value in distribution.values() if value is not None and value >= 0]
                competitiveness_dict[asset_class] = AssetSegmentation.segment_competitiveness(incentive_distribution)

        return competitiveness_dict

    def days_in_year_from_asset_category(self, asset_category: TradePairCategory) -> int:

        days_in_year = self.asset_breakdown.get(asset_category, {}).get("days_in_year")

        if days_in_year is None or days_in_year <= 0:
            raise ValueError(f"Days in year must be positive, instead of {days_in_year}")

        return days_in_year
