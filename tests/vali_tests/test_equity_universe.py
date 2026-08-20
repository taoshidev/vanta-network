"""
Guards for the bulk-added Russell 1000 equity-spot universe.

vali_config.py is the source of truth; runnable/generate_equity_universe.py is additive-only
(appends Russell tickers missing from the file, never edits/deletes existing ones). The CSV
(runnable/equity_universe/russell1000.csv) is just the bulk-add list. These tests check the CSV is
clean, the bulk-added members are valid EQUITIES/SPOT, dual-class display is dotted, and that the
generator's add-logic only touches genuinely-new tickers.
"""
import re
import unittest

from runnable.generate_equity_universe import (
    all_member_ids,
    current_generated_lines,
    read_clean_csv,
    render_member_line,
    tickers_to_add,
    CONFIG_PATH,
)
from vali_objects.vali_config import (
    InstrumentType,
    TradePairCategory,
)
from vali_objects.trade_pair import TRADE_PAIR_ID_TO_TRADE_PAIR


class TestEquityUniverse(unittest.TestCase):
    def setUp(self):
        self.clean_rows = read_clean_csv()
        self.config_text = open(CONFIG_PATH).read()

    def test_universe_csv_clean(self):
        symbols = [r["symbol"] for r in self.clean_rows]
        self.assertGreater(len(symbols), 900, "Russell 1000 universe unexpectedly small")
        self.assertEqual(len(symbols), len(set(symbols)), "duplicate symbols in universe CSV")
        for s in symbols:
            self.assertRegex(s, r"^[A-Za-z_][A-Za-z0-9_]*$", f"{s} is not a valid TradePair member name")

    def test_bulk_added_members_resolve_as_equity_spot(self):
        gen = [re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)", line).group(1) for line in current_generated_lines(self.config_text)]
        self.assertGreater(len(gen), 900, "bulk-added section unexpectedly small (did the generator run?)")
        for s in gen:
            tp = TRADE_PAIR_ID_TO_TRADE_PAIR.get(s)
            self.assertIsNotNone(tp, f"bulk-added member {s} does not resolve")
            self.assertEqual(tp.trade_pair_category, TradePairCategory.EQUITIES)
            self.assertEqual(tp.instrument_type, InstrumentType.SPOT)

    def test_dual_class_tickers_use_dotted_display(self):
        """Underscore ids (BF_A) expose the dotted vendor/display symbol (BF.A), like BRK_B/BRK.B."""
        gen = [re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)", line).group(1) for line in current_generated_lines(self.config_text)]
        dotted = [s for s in gen if "_" in s]
        self.assertTrue(dotted, "expected at least one dual-class underscore ticker (e.g. BF_A)")
        for s in dotted:
            self.assertEqual(TRADE_PAIR_ID_TO_TRADE_PAIR[s].trade_pair, s.replace("_", "."))

    def test_generator_is_additive(self):
        """Only genuinely-new tickers are added; ones already in the file are skipped (preserved)."""
        present = all_member_ids(self.config_text)
        # nothing to add when every CSV symbol already exists
        self.assertEqual(tickers_to_add(self.clean_rows, present), [])
        # a synthetic new ticker would be added with default values; an existing one would not
        rows = [{"symbol": "AAPL"}, {"symbol": "ZZNEWCO"}]
        self.assertEqual(tickers_to_add(rows, present), ["ZZNEWCO"])
        line = render_member_line("ZZNEWCO", "0.00009", "1.0")
        self.assertIn('["ZZNEWCO", "ZZNEWCO", 0.00009,', line)
        self.assertIn("SubaccountTierBaseLeverage(1.0)]", line)


if __name__ == "__main__":
    unittest.main()
