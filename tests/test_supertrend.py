import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import datetime
from strategy.supertrend import SupertrendStrategy

class TestSupertrendStrategy(unittest.TestCase):
    def setUp(self):
        self.config = {
            "index_symbol": "NSE:NIFTY 50",
            "exchange": "NFO",
            "st_period": 10,
            "st_multiplier": 3.0,
            "atr_period": 14,
            "capital": 1000000,
            "risk_per_trade": 0.01,
            "stop_loss_pct": 0.20,
            "lot_size": 75
        }
        self.broker = MagicMock()
        # Mock download_instruments and get_instruments
        self.broker.download_instruments.return_value = None
        self.broker.get_instruments.return_value = []

        self.strategy = SupertrendStrategy(self.config, self.broker)

    def test_calculate_indicators(self):
        # Create some dummy data
        data = {
            'high': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 112, 111, 113, 115],
            'low': [98, 100, 99, 101, 103, 102, 104, 106, 105, 107, 108, 110, 109, 111, 113],
            'close': [99, 101, 100, 102, 104, 103, 105, 107, 106, 108, 109, 111, 110, 112, 114],
            'volume': [1000] * 15
        }
        df = pd.DataFrame(data)

        # We need at least atr_period (14) + 1 for indicators
        df_out = self.strategy.calculate_indicators(df)

        self.assertIn('atr', df_out.columns)
        self.assertIn('st', df_out.columns)
        self.assertIn('is_uptrend', df_out.columns)

        # Verify Supertrend logic briefly
        last_row = df_out.iloc[-1]
        self.assertTrue(last_row['is_uptrend']) # Price is generally rising in dummy data

    def test_get_market_state(self):
        data = {
            'high': [100]*20,
            'low': [90]*20,
            'close': [95]*20,
            'volume': [1000]*20
        }
        df = pd.DataFrame(data)
        df = self.strategy.calculate_indicators(df)

        # With flat data, ATR will be constant, atr_rising will be False
        state = self.strategy.get_market_state(df)
        self.assertIn("calm", state.lower())


    def test_calculate_quantity(self):
        # 1% of 1M is 10k risk.
        # Premium 100, 20% SL is 20 points.
        # 10k / 20 = 500 units.
        # 500 rounded to lot size 75 is 450?
        # Wait, 500 // 75 = 6. 6 * 75 = 450.
        qty = self.strategy.calculate_quantity(100, False)
        self.assertEqual(qty, 450)

        # Volatile mult is 0.75. 450 * 0.75 = 337.5.
        # 337.5 // 75 = 4. 4 * 75 = 300.
        qty_v = self.strategy.calculate_quantity(100, True)
        self.assertEqual(qty_v, 300)

if __name__ == '__main__':
    unittest.main()
