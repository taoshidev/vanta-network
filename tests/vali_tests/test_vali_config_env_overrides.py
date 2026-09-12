"""
ValiConfig testnet knobs: the pro promotion criteria and transition grace period are read from the
environment once at import. Each case imports ValiConfig in a fresh interpreter so the override is
exercised exactly the way a validator process sees it.
"""
import json
import os
import subprocess
import sys
import unittest

from vali_objects import vali_config

OPT_IN = "PTN_ALLOW_CONFIG_OVERRIDES"

KNOBS = (
    OPT_IN,
    "PRO_CHALLENGE_MINIMUM_DAYS",
    "PRO_CHALLENGE_CALMAR_THRESHOLD",
    "PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD",
    "PRO_CHALLENGE_RETURNS_THRESHOLD_DEFAULT",
    "PRO_TRANSITION_GRACE_PERIOD_DAYS",
)

# Read the values through their consumers as well, so a copied literal anywhere would fail the test
_PROBE = """
import json
from vali_objects.vali_config import ValiConfig
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
print("PROBE " + json.dumps({
    "minimum_days": ValiConfig.PRO_CHALLENGE_MINIMUM_DAYS,
    "calmar": ValiConfig.PRO_CHALLENGE_CALMAR_THRESHOLD,
    "consistency": ValiConfig.PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD,
    "returns_default": ValiConfig.PRO_CHALLENGE_RETURNS_THRESHOLD_DEFAULT,
    "returns_by_class": sorted(set(ValiConfig.PRO_CHALLENGE_RETURNS_THRESHOLD.values())),
    "grace_days": ValiConfig.PRO_TRANSITION_GRACE_PERIOD_DAYS,
    "grace_ms": ValiConfig.PRO_TRANSITION_GRACE_PERIOD_MS,
    "bucket_minimum_days": MinerBucket.PRO_CHALLENGE_DIRECT.minimum_trading_days,
    "bucket_calmar": MinerBucket.PRO_CHALLENGE_DIRECT.calmar_threshold,
    "bucket_consistency": MinerBucket.PRO_CHALLENGE_DIRECT.daily_consistency_threshold,
    "bucket_returns": MinerBucket.PRO_CHALLENGE_DIRECT.returns_threshold(MinerAssetClass.FOREX),
    "bucket_grace_ms": MinerBucket.PRO_CHALLENGE_TRANSITION.grace_period_ms,
}))
"""


def _import_with_env(overrides: dict, opt_in: bool = True) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    if overrides and opt_in:
        env[OPT_IN] = "1"
    env.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=vali_config.BASE_DIR, env=env, capture_output=True, text=True, timeout=120,
    )


def _probe(overrides: dict) -> dict:
    proc = _import_with_env(overrides)
    assert proc.returncode == 0, proc.stderr
    line = [l for l in proc.stdout.splitlines() if l.startswith("PROBE ")][-1]
    return json.loads(line[len("PROBE "):])


class TestProConfigEnvOverrides(unittest.TestCase):

    def test_defaults_when_unset(self):
        values = _probe({})
        self.assertEqual(values["minimum_days"], 90)
        self.assertEqual(values["calmar"], 1.75)
        self.assertEqual(values["consistency"], 0.2)
        self.assertEqual(values["returns_default"], 0.06)
        self.assertEqual(values["returns_by_class"], [0.06])
        self.assertEqual(values["grace_days"], 7)
        self.assertEqual(values["grace_ms"], 7 * vali_config.ValiConfig.DAILY_MS)

    def test_env_overrides_reach_every_consumer(self):
        values = _probe({
            "PRO_CHALLENGE_MINIMUM_DAYS": "3",
            "PRO_CHALLENGE_CALMAR_THRESHOLD": "0.5",
            "PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD": "0.9",
            "PRO_CHALLENGE_RETURNS_THRESHOLD_DEFAULT": "0.01",
            "PRO_TRANSITION_GRACE_PERIOD_DAYS": "0.25",
        })
        self.assertEqual(values["minimum_days"], 3)
        self.assertEqual(values["bucket_minimum_days"], 3)
        self.assertEqual(values["calmar"], 0.5)
        self.assertEqual(values["bucket_calmar"], 0.5)
        self.assertEqual(values["consistency"], 0.9)
        self.assertEqual(values["bucket_consistency"], 0.9)
        self.assertEqual(values["returns_default"], 0.01)
        self.assertEqual(values["returns_by_class"], [0.01])
        self.assertEqual(values["bucket_returns"], 0.01)
        self.assertEqual(values["grace_days"], 0.25)
        self.assertEqual(values["grace_ms"], int(0.25 * vali_config.ValiConfig.DAILY_MS))
        self.assertEqual(values["bucket_grace_ms"], values["grace_ms"])

    def test_override_logs_a_warning(self):
        proc = _import_with_env({"PRO_CHALLENGE_MINIMUM_DAYS": "3"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PRO_CHALLENGE_MINIMUM_DAYS overridden by environment: 3 (default 90)", proc.stderr + proc.stdout)

    def test_knob_without_the_opt_in_flag_fails_import(self):
        """A stray knob must never quietly change a validator: it is a startup failure."""
        proc = _import_with_env({"PRO_CHALLENGE_MINIMUM_DAYS": "3"}, opt_in=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PTN_ALLOW_CONFIG_OVERRIDES=1 is not", proc.stderr)

    def test_blank_override_keeps_default(self):
        self.assertEqual(_probe({"PRO_CHALLENGE_CALMAR_THRESHOLD": ""})["calmar"], 1.75)

    def test_non_positive_or_non_numeric_override_fails_import(self):
        for knob, raw in (
            ("PRO_CHALLENGE_MINIMUM_DAYS", "0"),
            ("PRO_CHALLENGE_CALMAR_THRESHOLD", "-1"),
            ("PRO_TRANSITION_GRACE_PERIOD_DAYS", "week"),
            ("PRO_CHALLENGE_MINIMUM_DAYS", "1.5"),
            ("PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD", "inf"),
        ):
            with self.subTest(knob=knob, raw=raw):
                proc = _import_with_env({knob: raw})
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(f"{knob}={raw!r} must be a positive number", proc.stderr)


if __name__ == '__main__':
    unittest.main()
