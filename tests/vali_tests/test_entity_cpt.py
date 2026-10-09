"""
CPT schedule for entity subaccounts: registration, pro promotion and margin, in theta.

Each test in TestCptValuesRedesignTable is one row of the CPT Values Redesign table, which assumes a
$100K standard account and a $500K pro account. "Deducted" rows are one-off fees; "held" rows are the
additional margin an account carries on entering that state.
"""
import unittest

from vali_objects.vali_config import ValiConfig

S = 100_000
P = 500_000
FIVE = 0.05
THREE = 0.03


def std_reg_theta(size, dll=None):
    return size / ValiConfig.std_reg_cpt(size, dll)


def instant_reg_theta(size, eod=None):
    return size / ValiConfig.instant_reg_cpt(size, eod)


def margin(size, pro_funded=False, multiplier=1.0):
    # The slash ceiling: 5% of the account in every earning bucket
    return ValiConfig.margin_theta(size * 0.05, pro_funded, multiplier)


class TestCptValuesRedesignTable(unittest.TestCase):

    def test_standard_challenge_5pct_dll(self):
        self.assertAlmostEqual(std_reg_theta(S, FIVE), 66.67, places=2)

    def test_standard_challenge_3pct_dll(self):
        self.assertAlmostEqual(std_reg_theta(S, THREE), 40.0, places=2)

    def test_standard_funded_holds_5pct_margin_at_either_dll(self):
        # Margin is 5% of the account for every account, whatever its daily loss limit
        self.assertAlmostEqual(margin(S), 142.86, places=2)

    def test_grow_5pct_dll(self):
        self.assertAlmostEqual(ValiConfig.promotion_fee_theta(P, S, FIVE), 100.0, places=2)
        # Double payouts hold a second standard margin on top of the one already held
        self.assertAlmostEqual(margin(S, multiplier=2.0) - margin(S), 142.86, places=2)

    def test_grow_3pct_dll(self):
        self.assertAlmostEqual(ValiConfig.promotion_fee_theta(P, S, THREE), 60.0, places=2)
        self.assertAlmostEqual(margin(S, multiplier=2.0) - margin(S), 142.86, places=2)

    def test_pro_challenge_from_standard_challenge_5pct_dll(self):
        self.assertAlmostEqual(ValiConfig.promotion_fee_theta(P, S, FIVE), 100.0, places=2)

    def test_pro_challenge_from_standard_challenge_3pct_dll(self):
        self.assertAlmostEqual(ValiConfig.promotion_fee_theta(P, S, THREE), 60.0, places=2)

    def test_pro_funded_from_grow(self):
        # The same at 5% and 3% DLL: margin is 5% either way
        self.assertAlmostEqual(margin(P, pro_funded=True) - margin(S, multiplier=2.0), 71.43, places=2)

    def test_pro_funded_from_pro_challenge(self):
        self.assertAlmostEqual(margin(P, pro_funded=True), 357.14, places=2)

    def test_instant_funded_not_eligible_5pct_eod(self):
        self.assertAlmostEqual(instant_reg_theta(S, 0.05), 250.0, places=2)
        self.assertAlmostEqual(margin(S), 142.86, places=2)

    def test_instant_funded_not_eligible_8pct_eod(self):
        self.assertAlmostEqual(instant_reg_theta(S, 0.08), 333.33, places=2)
        self.assertAlmostEqual(margin(S), 142.86, places=2)

    def test_instant_funded_eligible_5pct_eod(self):
        # Pays the promotion fee of a 3% DLL standard account on top of the Instant Funded fee
        fee = instant_reg_theta(S, 0.05) + ValiConfig.promotion_fee_theta(P, S, THREE)
        self.assertAlmostEqual(fee, 310.0, places=2)
        self.assertAlmostEqual(margin(S), 142.86, places=2)

    def test_instant_funded_eligible_8pct_eod(self):
        fee = instant_reg_theta(S, 0.08) + ValiConfig.promotion_fee_theta(P, S, THREE)
        self.assertAlmostEqual(fee, 393.33, places=2)
        self.assertAlmostEqual(margin(S), 142.86, places=2)

    def test_pro_funded_from_instant_funded(self):
        self.assertAlmostEqual(margin(P, pro_funded=True) - margin(S), 214.29, places=2)


class TestRegistrationCpt(unittest.TestCase):

    def test_defaults(self):
        """Omitted thresholds price at a 5% daily loss limit and an 8% EOD high-water mark."""
        self.assertEqual(ValiConfig.std_reg_cpt(S), ValiConfig.std_reg_cpt(S, FIVE))
        self.assertEqual(ValiConfig.instant_reg_cpt(S), ValiConfig.instant_reg_cpt(S, 0.08))

    def test_halved_at_or_below_10k(self):
        small = ValiConfig.REG_CPT_HALVING_THRESHOLD
        self.assertEqual(ValiConfig.std_reg_cpt(small, FIVE), 750)
        self.assertEqual(ValiConfig.std_reg_cpt(small, THREE), 1250)
        self.assertEqual(ValiConfig.instant_reg_cpt(small, 0.05), 200)
        self.assertEqual(ValiConfig.instant_reg_cpt(small, 0.08), 150)
        self.assertAlmostEqual(std_reg_theta(small, FIVE), 13.33, places=2)
        self.assertAlmostEqual(instant_reg_theta(small, 0.08), 66.67, places=2)

    def test_full_rate_above_10k(self):
        just_over = ValiConfig.REG_CPT_HALVING_THRESHOLD + 1
        self.assertEqual(ValiConfig.std_reg_cpt(just_over, FIVE), 1500)
        self.assertEqual(ValiConfig.instant_reg_cpt(just_over, 0.05), 400)

    def test_an_unknown_threshold_prices_at_the_default(self):
        """Creation only stores listed values, so an unknown one is corrupt data: priced at the default
        rather than raising inside a promotion or a peer's sync."""
        self.assertEqual(ValiConfig.std_reg_cpt(S, 0.04), ValiConfig.std_reg_cpt(S, FIVE))
        self.assertEqual(ValiConfig.instant_reg_cpt(S, 0.03), ValiConfig.instant_reg_cpt(S, 0.08))
        self.assertAlmostEqual(ValiConfig.promotion_fee_theta(P, S, 0.04), ValiConfig.promotion_fee_theta(P, S, FIVE))


class TestPromotionFee(unittest.TestCase):

    def test_free_at_twice_the_standard_size(self):
        for dll in (FIVE, THREE, None):
            with self.subTest(dll=dll):
                self.assertAlmostEqual(ValiConfig.promotion_fee_theta(2 * S, S, dll), 0.0)

    def test_never_negative_below_twice_the_standard_size(self):
        for pro_size in (0.0, S, 1.5 * S):
            with self.subTest(pro_size=pro_size):
                self.assertEqual(ValiConfig.promotion_fee_theta(pro_size, S, FIVE), 0.0)

    def test_small_account_credit_is_capped_at_the_full_rate(self):
        """A <=10k account paid the halved rate to register, but is only credited the full-rate amount,
        so its promotion is free at exactly twice its size and charged above it."""
        small = ValiConfig.REG_CPT_HALVING_THRESHOLD
        self.assertAlmostEqual(ValiConfig.promotion_fee_theta(2 * small, small, FIVE), 0.0)
        self.assertAlmostEqual(ValiConfig.promotion_fee_theta(4 * small, small, FIVE), 6.67, places=2)

    def test_pro_registration_is_not_halved(self):
        self.assertAlmostEqual(ValiConfig.promotion_fee_theta(P, 0.0, FIVE), P / 3000)


class TestMarginTheta(unittest.TestCase):

    def test_pro_funded_uses_the_pro_margin_cpt(self):
        self.assertAlmostEqual(ValiConfig.margin_theta(7_000, pro_funded=True), 100.0)
        self.assertAlmostEqual(ValiConfig.margin_theta(7_000), 200.0)

    def test_never_negative(self):
        self.assertEqual(ValiConfig.margin_theta(-1.0), 0.0)
        self.assertEqual(ValiConfig.margin_theta(None), 0.0)


if __name__ == '__main__':
    unittest.main()
