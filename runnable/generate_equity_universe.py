#!/usr/bin/env python3
"""
Bulk-add Russell 1000 equity-spot TradePair members into trade_pair.py.

SOURCE OF TRUTH IS trade_pair.py, NOT THE CSV
----------------------------------------------
The validator runs off the TradePair enum in trade_pair.py, so that file is the source of
truth for membership AND per-pair values. This generator is ADDITIVE-ONLY: it appends Russell
tickers that are not already present, and it NEVER edits or deletes anything already in the
file. So you may freely hand-add, hand-remove, or re-tune any member in trade_pair.py and a
re-run will not clobber it. The CSV's only job is to let us bulk-add ~1000 tickers quickly.

  * Per-pair fees / SubaccountTierBaseLeverage are written as LITERALS (not shared constants)
    so each pair can be tuned individually in trade_pair.py.
  * The ExposureGroup (GICS sector) is written from the CSV's `sector` column when a ticker is
    first added, and is never rewritten afterward — hand-tuned sectors survive a re-run. Use
    --check to report tickers whose CSV sector has since diverged from trade_pair.py.
  * Re-running only inserts tickers missing from the file; existing members are left byte-for-byte.
  * To REMOVE a ticker: drop it from the CSV (so it won't be re-added) and, if it has positions,
    block it via BLOCKED_TRADE_PAIR_IDS — never hard-delete a member (trade_pair_id is an
    immutable on-disk position key). Hand-deleting a member line that is still in the CSV will
    be re-added on the next run.

WHERE THE CSV COMES FROM
------------------------
iShares Russell 1000 ETF (IWB) fully replicates the index, so its equity holdings are a
complete proxy for membership (the authoritative source is FTSE Russell's licensed file).
  1. https://www.ishares.com/us/products/239707/ishares-russell-1000-etf
  2. Holdings -> "Detailed Holdings and Analytics" -> download the CSV.
  3. Save to the repo root as IWB_holdings.csv (gitignored; the RAW file is not committed).
Only `Asset Class == "Equity"` rows are constituents. FTSE Russell reconstitutes semi-annually
from 2026 (June + the 2nd Friday of December; first December run Dec 11, 2026).

USAGE
-----
  python runnable/generate_equity_universe.py --raw IWB_holdings.csv   # refresh universe CSV + add new tickers
  python runnable/generate_equity_universe.py                          # add new tickers from existing CSV
  python runnable/generate_equity_universe.py --check                  # report which CSV tickers aren't in config yet
"""
import argparse
import csv
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "vali_objects", "trade_pair.py")
CLEAN_CSV_PATH = os.path.join(REPO_ROOT, "runnable", "equity_universe", "russell1000.csv")

# The generated tickers live in their own section so the generator knows where to append new ones.
# HEADER_MARKER is a stable substring used to locate that section (so the header wording can change
# without breaking detection). ANCHOR (the next real section comment) closes it.
HEADER_MARKER = "Russell 1000 stocks"
HEADER = [
    "    # Russell 1000 stocks bulk-added by runnable/generate_equity_universe.py (additive: appends new",
    "    # tickers, never touches existing). Per-pair fees/base literals here are hand-editable.",
]
ANCHOR = "# Equities - Sector ETFs"

# IWB concatenates dual-class tickers that carry a dot on the exchange (BRK.B -> "BRKB"). Repo id
# convention is dot -> underscore (BRK_B); the vendor/display symbol (value[1]) keeps the dot,
# derived as id.replace("_", "."). Plain letter-suffix classes (GOOGL, FOXA, ...) need nothing.
# RECONSTITUTION CAVEAT: a NEW dotted-class ticker arrives from IWB concatenated (e.g. "XYZA" for
# XYZ.A) — a valid identifier — so without an entry here it would be bulk-added wrongly as "XYZA"
# and fail vendor lookups. Add new dotted-class tickers here (IWB form -> underscore id) before re-running.
IWB_TO_CANONICAL = {
    "BRKB": "BRK_B",    # Berkshire Hathaway B  (BRK.B)
    "BFA": "BF_A",      # Brown-Forman A        (BF.A)
    "BFB": "BF_B",      # Brown-Forman B        (BF.B)
    "LENB": "LEN_B",    # Lennar B              (LEN.B)
    "HEIA": "HEI_A",    # Heico A               (HEI.A)
    "UHALB": "UHAL_B",  # U-Haul Holding N      (UHAL.B)
}

_MEMBER_LINE = re.compile(r"\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\[")


def _member_region(config_text: str) -> tuple[int, int]:
    """[start, end) char offsets of the TradePair member block (class body up to first @property)."""
    start = config_text.index("class TradePair(Enum):")
    end = config_text.index("@property", start)
    return start, end


def _bounds(lines: list[str]) -> tuple[int | None, int]:
    """(generated-section start line index or None, ANCHOR line index)."""
    anchor_i = next(i for i, line in enumerate(lines) if ANCHOR in line)
    header_i = next((i for i, line in enumerate(lines) if HEADER_MARKER in line and i < anchor_i), None)
    return header_i, anchor_i


def all_member_ids(config_text: str) -> set[str]:
    """Every TradePair member id already defined in the file (curated + previously bulk-added)."""
    start, end = _member_region(config_text)
    return set(re.findall(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[", config_text[start:end], re.M))


def current_generated_lines(config_text: str) -> list[str]:
    """Member lines already in the generated section, kept verbatim (never rewritten)."""
    lines = config_text.split("\n")
    h, a = _bounds(lines)
    if h is None:
        return []
    return [line for line in lines[h + 1:a] if _MEMBER_LINE.match(line)]


def _members_text_excluding_generated(config_text: str) -> str:
    lines = config_text.split("\n")
    h, a = _bounds(lines)
    if h is None:
        start, end = _member_region(config_text)
        return config_text[start:end]
    return "\n".join(lines[:h] + lines[a:])


def default_fees_base(config_text: str) -> tuple[str, str]:
    """Default (fees, base) for new tickers — read from an existing equity-spot member, never hardcoded."""
    m = re.search(
        r'=\s*\[\s*"[^"]*",\s*"[^"]*",\s*([0-9.eE+-]+),\s*'
        r"EQUITIES_MIN_LEVERAGE,.*?TradePairCategory\.EQUITIES,\s*InstrumentType\.SPOT,\s*"
        r"SubaccountTierBaseLeverage\(([0-9.eE+-]+)\)(?:,\s*ExposureGroup\.[A-Z_]+)?\]",
        _members_text_excluding_generated(config_text),
    )
    if not m:
        raise RuntimeError("no reference EQUITIES/SPOT member found to read default fees/base")
    return m.group(1), m.group(2)


def exposure_group_members(config_text: str) -> set[str]:
    """ExposureGroup member names defined in trade_pair.py (parsed, not imported)."""
    block = re.search(r"class ExposureGroup\(str, Enum\):\n(.*?)(?=\n\S)", config_text, re.S)
    if not block:
        raise RuntimeError("no ExposureGroup enum found in trade_pair.py")
    return set(re.findall(r"^\s+([A-Z][A-Z_]*)\s*=", block.group(1), re.M))


def sector_to_member(sector: str) -> str:
    """CSV sector label -> ExposureGroup member name. Values are the labels verbatim."""
    return sector.strip().upper().replace(" ", "_")


def clean_iwb_holdings(raw_path: str) -> list[dict]:
    """Raw IWB export -> sorted list of {symbol, name, sector} for every Equity constituent."""
    with open(raw_path, newline="", encoding="utf-8-sig") as f:
        lines = f.readlines()
    hdr = next(i for i, line in enumerate(lines) if line.startswith("Ticker,"))
    rows = {}
    for r in csv.DictReader(lines[hdr:]):
        if (r.get("Asset Class") or "").strip() != "Equity":
            continue
        ticker = (r.get("Ticker") or "").strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", ticker) and ticker not in IWB_TO_CANONICAL:
            continue  # skip cash placeholder rows like "-"
        symbol = IWB_TO_CANONICAL.get(ticker, ticker)
        rows[symbol] = {"symbol": symbol, "name": (r.get("Name") or "").strip(), "sector": (r.get("Sector") or "").strip()}
    return sorted(rows.values(), key=lambda x: x["symbol"])


def write_clean_csv(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(CLEAN_CSV_PATH), exist_ok=True)
    with open(CLEAN_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")  # LF, not csv's default CRLF (match the repo)
        w.writerow(["symbol", "name", "sector"])
        for r in rows:
            w.writerow([r["symbol"], r["name"], r["sector"]])


def read_clean_csv() -> list[dict]:
    with open(CLEAN_CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def render_member_line(symbol: str, fees: str, base: str, sector: str) -> str:
    display = symbol.replace("_", ".")  # dotted vendor/display symbol (BRK_B -> BRK.B)
    return (
        f'    {symbol} = ["{symbol}", "{display}", {fees}, '
        f"EQUITIES_MIN_LEVERAGE, EQUITIES_MAX_LEVERAGE, "
        f"TradePairCategory.EQUITIES, InstrumentType.SPOT, SubaccountTierBaseLeverage({base}), "
        f"ExposureGroup.{sector_to_member(sector)}]"
    )


def config_sectors(config_text: str) -> dict[str, str]:
    """Member id -> ExposureGroup member name, as currently written in trade_pair.py."""
    start, end = _member_region(config_text)
    return dict(re.findall(
        r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[.*?ExposureGroup\.([A-Z_]+)\]",
        config_text[start:end], re.M,
    ))


def sector_drift(clean_rows: list[dict], config_text: str) -> list[tuple[str, str, str]]:
    """(symbol, csv sector, config sector) where the CSV disagrees with trade_pair.py.

    Additive-only means sectors are never rewritten, so drift is reported, not corrected.
    Deliberate hand-overrides (UBER, APP, ...) show up here permanently and by design.
    """
    in_config = config_sectors(config_text)
    drift = []
    for r in clean_rows:
        current = in_config.get(r["symbol"])
        wanted = sector_to_member(r.get("sector") or "")
        if current and wanted and current != wanted:
            drift.append((r["symbol"], wanted, current))
    return drift


def tickers_to_add(clean_rows: list[dict], present_ids: set[str]) -> list[str]:
    """CSV symbols not already defined anywhere in the file (additive: only genuinely-new tickers)."""
    return [r["symbol"] for r in clean_rows if r["symbol"] not in present_ids]


def splice_into_config(config_text: str, kept_lines: list[str], new_lines: list[str]) -> str:
    lines = config_text.split("\n")
    h, a = _bounds(lines)
    block = HEADER + kept_lines + new_lines + [""]
    new = (lines[:h] if h is not None else lines[:a]) + block + lines[a:]
    return "\n".join(new)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", help="Path to raw IWB_holdings.csv; refreshes the committed universe CSV from it.")
    ap.add_argument("--check", action="store_true", help="Report CSV tickers not yet in trade_pair.py; exit 0.")
    args = ap.parse_args()

    config_text = open(CONFIG_PATH).read()

    if args.raw:
        rows = clean_iwb_holdings(args.raw)
        write_clean_csv(rows)
        print(f"Cleaned {len(rows)} Russell 1000 equities -> {os.path.relpath(CLEAN_CSV_PATH, REPO_ROOT)}")

    clean_rows = read_clean_csv()
    present = all_member_ids(config_text)
    to_add = tickers_to_add(clean_rows, present)

    if args.check:
        print(f"{len(to_add)} CSV ticker(s) not yet in trade_pair.py" + (f": {to_add[:20]}{' ...' if len(to_add) > 20 else ''}" if to_add else "."))
        drift = sector_drift(clean_rows, config_text)
        if drift:
            print(f"{len(drift)} ticker(s) whose CSV sector differs from trade_pair.py:")
            for symbol, csv_sector, config_sector in drift:
                print(f"  {symbol:8} {csv_sector} (csv)  !=  {config_sector} (config)")
            print("(additive-only: sectors are never rewritten — hand-edit if desired)")
        else:
            print("No sector drift between the CSV and trade_pair.py.")
        return 0

    known_groups = exposure_group_members(config_text)
    sector_by_symbol = {r["symbol"]: (r.get("sector") or "").strip() for r in clean_rows}
    for sym in to_add:
        sector = sector_by_symbol[sym]
        if not sector:
            raise RuntimeError(f"{sym} has no sector in the CSV; cannot emit an ExposureGroup")
        if sector_to_member(sector) not in known_groups:
            raise RuntimeError(
                f"{sym}: sector {sector!r} has no ExposureGroup member "
                f"(expected {sector_to_member(sector)}). Add it to trade_pair.py first."
            )

    df, db = default_fees_base(config_text)
    new_lines = [render_member_line(sym, df, db, sector_by_symbol[sym]) for sym in to_add]
    kept_lines = current_generated_lines(config_text)
    new_text = splice_into_config(config_text, kept_lines, new_lines)
    if new_text != config_text:
        with open(CONFIG_PATH, "w") as f:
            f.write(new_text)
    print(f"Added {len(new_lines)} new ticker(s); {len(kept_lines)} existing bulk-added members left untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
