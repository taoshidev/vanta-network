# Pro leverage: unresolved items

Where the implemented pro leverage tables (`ValiConfig.PRO_*`, resolved by
`leverage_utils.get_pro_positional_leverage`) differ from the pro launch spec, and what was
assumed in the meantime. Every item here needs a product decision; none is a code bug.

The spec appendix listing the non-tradable equity names is now available (2026 August quarterly
review) and item 3 below has been reconciled against it directly.

## 1. BCHUSDC has no leverage in the spec

The spec lists **15** pro crypto pairs (5x: BTC, ETH, SOL, XRP, DOGE / 2x: HYPE, SUI, BNB /
1.5x: kPEPE, ADA, ZEC, LINK / 1x: LTC, AVAX, TRX) plus a separate not-tradable list of **16**
(3 removed: PAXG, TON, XMR; 13 below the 1x capacity floor: AAVE, ALGO, ARB, ASTER, CRV, DOT,
ENA, NEAR, PUMP, TAO, UNI, WLD, ZRO). Those two lists total the spec's full 31-pair crypto
universe. `BCHUSDC` appears in neither list, yet the code flags it `TradePair.is_pro = True`,
giving the code's pro universe **16** tradable pairs against the spec's 15. This is confirmed
against the full spec text, not just the leverage table — BCH is not an oversight in a partial
list, the spec is silent on it entirely.

**Assumed:** `BCHUSDC` falls through to `ValiConfig.PRO_DEFAULT_POSITIONAL_LEVERAGE` = **1.0x**,
the tightest value in the crypto table, so it is never accidentally granted more room than
intended.

**Decision needed:** give BCH an explicit leverage, or drop it from the pro universe by setting
`is_pro = False` on `TradePair.BCHUSDC`.

## 2. Equities: 20 Hyperliquid duplicate perps — resolved, excluded

Was 753 pro equity names in code vs 733 in the spec. The 20-name gap was the Hyperliquid-sourced
equity perps, each a duplicate listing of a Vanta-sourced name already in the set:

```
AAPLUSDC  AMDUSDC   AMZNUSDC  COINUSDC  CRCLUSDC  GOOGLUSDC  HOODUSDC
INTCUSDC  METAUSDC  MSFTUSDC  MSTRUSDC  MUUSDC    NFLXUSDC   NVDAUSDC
ORCLUSDC  PLTRUSDC  SNDKUSDC  SPCXUSDC  TSLAUSDC  TSMUSDC
```

**Resolved:** set `is_pro = False` on all 20 (`trade_pair.py`, "Equity perp futures" block).
Vanta-sourced pro equities remain **exactly 733**, matching the spec's 714 US stocks + 19 ETFs,
and the HL duplicates no longer inflate the pro universe. Leverage was unaffected either way (all
equities are 2x) — this was purely a universe-membership question, settled by the spec's own
wording ("duplicate listings ... are excluded") and `entity_miner.md`'s claim that pro accounts
are Vanta-native.

## 3. Appendix reconciled — exact match

Resolved, now that the appendix text is available. The code has exactly **305** Vanta-sourced
equities with `is_pro = False` out of **1038** Vanta equities total (733 pro), and that 305-name
set is identical to the spec's not-tradable appendix (3 duplicate ETFs, 13 ETFs and 289 stocks
below the $100M ADV floor), modulo the `.`/`_` share-class formatting difference between the
spec's tickers (e.g. `BF.A`) and `TradePair.trade_pair_id` (`BF_A`). No action needed.

## 4. `entity_miner.md` claim that pro is Vanta-native only is wrong

The Account Types section states pro accounts "are Vanta-native only: they trade Vanta-sourced
pairs (forex and equities) and cannot trade Hyperliquid-sourced pairs". The actual pro universe
is 786 pairs: **761 Vanta-sourced, 25 Hyperliquid-sourced** — all 16 crypto, all 6 commodities
and all 3 indices are HL-sourced. (The 20 HL equity perps that used to also count here were
excluded in item 2, so the equities side of the pro universe is now Vanta-only.)

The new spec grants explicit leverage to crypto, commodities and indices, so the restriction is
clearly no longer intended. The sentence was corrected when the pro leverage tables landed; it is
recorded here because it also affects fee selection: `position.py` gates the pro fee schedule on
`use_pro = self.is_pro and src == VANTA`, so a pro account holding any of those 25 HL-sourced
pairs pays the **standard** carry/borrow schedule, not the pro one. That is correct today only
because the pro and standard rates are identical clones (`trade_pair.py:173-175`).

**Decision needed:** confirm that HL-sourced pro positions should keep paying HL funding plus the
standard schedule once the pro rates diverge from standard.
