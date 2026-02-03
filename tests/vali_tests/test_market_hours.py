import time
import unittest
from datetime import datetime, timedelta, timezone

from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_objects.vali_config import TradePair


class TestMarketHours(unittest.TestCase):

    def setUp(self):
        self.umc = UnifiedMarketCalendar()

    def test_forex_success(self):
        # Noon UTC on a Wednesday, assuming forex markets are open
        timestamp = TimeUtil.timestamp_to_millis(datetime(2023, 5, 3, 12, 0, tzinfo=timezone.utc))
        trade_pair = TradePair.EURUSD  # Assuming this enum exists
        self.assertTrue(self.umc.is_market_open(trade_pair, timestamp))

    def test_equities_success(self):
         # 11 AM EST on a weekday, NYSE should be open
         timestamp = TimeUtil.timestamp_to_millis(datetime(2024, 5, 3, 15, 0, tzinfo=timezone.utc))  # 15:00 UTC is 11:00 EST
         trade_pair = TradePair.NVDA  # Assuming NYSE based and the enum exists
         self.assertTrue(self.umc.is_market_open(trade_pair, timestamp))

    def test_forex_fail_holiday_weekend(self):
        # Christmas, assuming forex markets are closed globally
        trade_pair = TradePair.EURUSD
        # Days when the market should be closed
        closed_days = [
            (datetime(2023, 1, 1, 12, 0), "New Year's Day"),  # New Year's Day (or nearest workday)
            (datetime(2023, 4, 7, 12, 0), "Good Friday"),  # Good Friday
            (datetime(2023, 12, 25, 12, 0), "Christmas Day"),  # Christmas Day
            (datetime(2023, 12, 26, 12, 0), "Boxing Day"),  # Boxing Day
        ]

        for day, description in closed_days:
            timestamp = TimeUtil.timestamp_to_millis(day.replace(tzinfo=timezone.utc))
            with self.subTest(day=description):
                self.assertFalse(self.umc.is_market_open(trade_pair, timestamp),
                                 f"Market should be closed on {description}")

    def test_forex_success_holidays(self):
        # Christmas, assuming forex markets are closed globally
        trade_pair = TradePair.EURUSD
        # Days when the market should be open
        open_days = [
            (datetime(2023, 7, 4, 15, 0), "July 4th"),  # Independence Day
            (datetime(2023, 10, 31, 12, 0), "Halloween"),  # Halloween is not a public holiday affecting markets
            (datetime(2023, 2, 14, 12, 0), "Valentine's Day"),  # Valentine's Day
            (datetime(2023, 3, 17, 12, 0), "St. Patrick's Day"),  # St. Patrick's Day
        ]

        for day, description in open_days:
            timestamp = TimeUtil.timestamp_to_millis(day.replace(tzinfo=timezone.utc))
            with self.subTest(day=description):
                self.assertTrue(self.umc.is_market_open(trade_pair, timestamp),
                                f"Market should be open on {description}")


    def test_equities_fail_holiday_weekend(self):
        # July 4th, a US market holiday, assuming markets are closed
        trade_pair = TradePair.MSFT
        timestamp = TimeUtil.timestamp_to_millis(datetime(2023, 7, 4, 15, 0, tzinfo=timezone.utc))
        self.assertFalse(self.umc.is_market_open(trade_pair, timestamp))
        timestamp = TimeUtil.timestamp_to_millis(datetime(2023, 12, 25, 0, 0, tzinfo=timezone.utc))  #Chirstimas
        self.assertFalse(self.umc.is_market_open(trade_pair, timestamp))
        timestamp = TimeUtil.timestamp_to_millis(datetime(2024, 12, 25, 0, 0, tzinfo=timezone.utc))  # Chirstimas
        self.assertFalse(self.umc.is_market_open(trade_pair, timestamp))

    def test_crypto_success(self):
        # Cryptos are always open; test any time
        timestamp = TimeUtil.timestamp_to_millis(datetime(2023, 12, 25, 0, 0, tzinfo=timezone.utc))  #Chirstimas
        trade_pair = TradePair.BTCUSD
        self.assertTrue(self.umc.is_market_open(trade_pair, timestamp))

    def test_forex_close_time_edge(self):
        # Test just before and just after the typical forex market close time (Friday 21:00 UTC)
        friday_close_before = TimeUtil.timestamp_to_millis(datetime(2023, 5, 5, 20, 59, tzinfo=timezone.utc))
        friday_close_after = TimeUtil.timestamp_to_millis(datetime(2023, 5, 5, 21, 1, tzinfo=timezone.utc))
        trade_pair = TradePair.EURUSD
        self.assertTrue(self.umc.is_market_open(trade_pair, friday_close_before))
        self.assertFalse(self.umc.is_market_open(trade_pair, friday_close_after))

    def test_failure_invalid_trade_pair(self):
        # Test with an undefined trade pair
        timestamp = TimeUtil.timestamp_to_millis(datetime(2023, 5, 3, 12, 0, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            self.umc.is_market_open(None, timestamp)

    def test_forex_closed_on_major_holiday(self):
        # Test forex market closure on New Year's Day
        new_years_day = TimeUtil.timestamp_to_millis(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc))
        trade_pair = TradePair.EURUSD
        self.assertFalse(self.umc.is_market_open(trade_pair, new_years_day))

    def test_forex_open_after_holidays(self):
        # Test forex market re-opening post New Year's Day
        day_after_new_years = TimeUtil.timestamp_to_millis(datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc))
        trade_pair = TradePair.EURUSD
        self.assertTrue(self.umc.is_market_open(trade_pair, day_after_new_years))

    def test_forex_close_time_before_holiday(self):
        # Test just before the Christmas market closure for Forex
        before_christmas_close_forex = TimeUtil.timestamp_to_millis(datetime(2023, 12, 24, 23, 59, tzinfo=timezone.utc))
        trade_pair = TradePair.EURUSD
        self.assertTrue(self.umc.is_market_open(trade_pair, before_christmas_close_forex))

    def test_forex_open_time_after_holiday(self):
        # Assuming the market re-opens at 5:00 PM New York time, which is 22:00 UTC during Standard Time
        after_new_years_open_forex = TimeUtil.timestamp_to_millis(datetime(2024, 1, 2, 22, 1, tzinfo=timezone.utc))
        trade_pair = TradePair.EURUSD
        self.assertTrue(self.umc.is_market_open(trade_pair, after_new_years_open_forex))

    def test_weekly_market_half_hour_intervals(self):
        t0 = time.time()
        # Start from Monday 00:00 UTC
        start_datetime = datetime(2023, 5, 1, 0, 0, tzinfo=timezone.utc)  # Ensure this date is a Monday

        # Iterate through each half-hour of the week, total 336 half-hours (24 hours * 7 days * 2)
        for half_hour in range(336):
            current_datetime = start_datetime + timedelta(minutes=30 * half_hour)
            timestamp = TimeUtil.timestamp_to_millis(current_datetime)

            # Test behavior for Forex (EURUSD)
            if current_datetime.weekday() == 4:
                if current_datetime.hour >= 21:  # Friday after 21:00 UTC
                    self.assertFalse(self.umc.is_market_open(TradePair.EURUSD, timestamp))
                else:
                    self.assertTrue(self.umc.is_market_open(TradePair.EURUSD, timestamp))
            elif current_datetime.weekday() == 6:  # Sunday
                if current_datetime.hour < 21:
                    self.assertFalse(self.umc.is_market_open(TradePair.EURUSD, timestamp))
                else:
                    self.assertTrue(self.umc.is_market_open(TradePair.EURUSD, timestamp))
            elif current_datetime.weekday() == 5:  # Saturday
                self.assertFalse(self.umc.is_market_open(TradePair.EURUSD, timestamp))
            else:
                self.assertTrue(self.umc.is_market_open(TradePair.EURUSD, timestamp))

            # Test behavior for Equities (assuming NYSE hours roughly 13:30 to 20:00 UTC)
            if 0 <= current_datetime.weekday() <= 4:  # Monday to Friday
                if 13 <= current_datetime.hour < 20:
                    if current_datetime.hour == 13 and current_datetime.minute < 30:
                        self.assertFalse(self.umc.is_market_open(TradePair.MSFT, timestamp))
                        self.assertFalse(self.umc.is_market_open(TradePair.MSFT, timestamp))
                    else:
                        self.assertTrue(self.umc.is_market_open(TradePair.MSFT, timestamp))
                        self.assertTrue(self.umc.is_market_open(TradePair.NVDA, timestamp))
                else:
                    self.assertFalse(self.umc.is_market_open(TradePair.GOOGL, timestamp))
                    self.assertFalse(self.umc.is_market_open(TradePair.NVDA, timestamp))
            else:  # Saturday and Sunday
                self.assertFalse(self.umc.is_market_open(TradePair.AMZN, timestamp))
                self.assertFalse(self.umc.is_market_open(TradePair.AMZN, timestamp))
        print(f'Finished in {time.time() - t0}')


if __name__ == '__main__':
    unittest.main()
