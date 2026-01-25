import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import numpy as np
import datetime
import time
import logging
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
load_dotenv()
import mibian

from logger import logger
from brokers import BrokerGateway, OrderRequest, Exchange, OrderType, TransactionType, ProductType

class SupertrendStrategy:
    """
    Supertrend Options Trading Strategy

    Implements a Supertrend and ATR based strategy with market regime detection.
    """

    def __init__(self, config: Dict, broker: BrokerGateway, order_tracker=None):
        self.config = config
        self.broker = broker
        self.order_tracker = order_tracker

        # Load parameters
        self.index_symbol = config.get("index_symbol", "NSE:NIFTY 50")
        self.exchange = config.get("exchange", "NFO")
        self.st_period = config.get("st_period", 10)
        self.st_multiplier = config.get("st_multiplier", 3.0)
        self.atr_period = config.get("atr_period", 14)

        self.capital = config.get("capital", 1000000)
        self.risk_per_trade = config.get("risk_per_trade", 0.01)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.20)
        self.trail_atr_mult = config.get("trail_atr_mult", 1.5)
        self.profit_threshold_pct = config.get("profit_threshold_pct", 0.30)

        self.product_type = ProductType.MARGIN if config.get("product_type", "NRML") == "NRML" else ProductType.INTRADAY
        self.tag = config.get("tag", "SUPERTREND_STRAT")
        self.lot_size = config.get("lot_size", 65)

        self.min_delta = config.get("min_delta", 0.30)
        self.max_delta = config.get("max_delta", 0.45)
        self.volatile_size_mult = config.get("volatile_size_mult", 0.75)

        # Timezone handling for India
        self.tz = ZoneInfo("Asia/Kolkata")

        self.start_trading_time = datetime.datetime.strptime(config.get("start_trading_time", "09:40"), "%H:%M").time()
        self.no_new_trades_after = datetime.datetime.strptime(config.get("no_new_trades_after", "14:30"), "%H:%M").time()
        self.last_exit_time = datetime.datetime.strptime(config.get("last_exit_time", "15:00"), "%H:%M").time()
        self.square_off_time = datetime.datetime.strptime(config.get("square_off_time", "15:15"), "%H:%M").time()

        self.min_time_gap_candles = config.get("min_time_gap_candles", 3)
        self.require_volume_confirmation = config.get("require_volume_confirmation", True)
        self.volume_period = config.get("volume_period", 20)

        # Internal state
        self.current_position: Optional[Dict] = None # Stores info about active trade
        self.last_flip_index = -1
        self.last_signal = None # 'BUY' or 'SELL'
        self.daily_loss = 0
        self.max_daily_loss = self.capital * 0.02

        # Setup broker
        logger.info("Initializing Supertrend Strategy...")
        self.broker.download_instruments()
        self.all_instruments = self.broker.get_instruments()

        # For Greeks
        self.interest_rate = config.get("interest_rate", 10.0)
        self.todays_volatility = config.get("todays_volatility", 20.0)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates ATR and Supertrend"""
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['tr'] = np.max(ranges, axis=1)
        df['atr'] = df['tr'].rolling(self.atr_period).mean()

        # Supertrend
        hl2 = (df['high'] + df['low']) / 2
        df['upperband'] = hl2 + (self.st_multiplier * df['atr'])
        df['lowerband'] = hl2 - (self.st_multiplier * df['atr'])
        df['is_uptrend'] = True

        for i in range(1, len(df)):
            if pd.isna(df['upperband'].iloc[i-1]) or pd.isna(df['lowerband'].iloc[i-1]):
                continue
            if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
                df.at[df.index[i], 'is_uptrend'] = True
            elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
                df.at[df.index[i], 'is_uptrend'] = False
            else:
                df.at[df.index[i], 'is_uptrend'] = df['is_uptrend'].iloc[i-1]

                if df['is_uptrend'].iloc[i] and df['lowerband'].iloc[i] < df['lowerband'].iloc[i-1]:
                    df.at[df.index[i], 'lowerband'] = df['lowerband'].iloc[i-1]
                if not df['is_uptrend'].iloc[i] and df['upperband'].iloc[i] > df['upperband'].iloc[i-1]:
                    df.at[df.index[i], 'upperband'] = df['upperband'].iloc[i-1]

        df['st'] = np.where(df['is_uptrend'], df['lowerband'], df['upperband'])

        # ATR Direction
        df['atr_rising'] = df['atr'] > df['atr'].shift(1)

        # Volume Average
        if self.require_volume_confirmation:
            df['vol_avg'] = df['volume'].rolling(self.volume_period).mean()
            df['vol_ok'] = df['volume'] > df['vol_avg']
        else:
            df['vol_ok'] = True

        return df

    def get_market_state(self, df: pd.DataFrame) -> str:
        """Determines market state based on last candle"""
        last_candle = df.iloc[-1]
        is_uptrend = last_candle['is_uptrend']
        atr_rising = last_candle['atr_rising']
        current_index = len(df) - 1

        # Simple flip check: did it just flip?
        if len(df) > 1:
            prev_candle = df.iloc[-2]
            if prev_candle['is_uptrend'] != is_uptrend:
                # Direction changed
                self.last_flip_index = current_index

                # Check for confirmation: last candle close beyond ST line
                if is_uptrend and last_candle['close'] <= last_candle['st']:
                    return "Choppy"
                if not is_uptrend and last_candle['close'] >= last_candle['st']:
                    return "Choppy"

        # Anti-whipsaw: Minimum time gap between flips
        if self.last_flip_index != -1 and (current_index - self.last_flip_index) < self.min_time_gap_candles:
            return "Choppy"

        if is_uptrend:
            return "Bullish volatile" if atr_rising else "Bullish calm"
        else:
            return "Bearish volatile" if atr_rising else "Bearish calm"

    def select_strike(self, option_type: str, spot_price: float) -> Optional[Dict]:
        """Selects near-ATM strike based on delta"""
        expiry_date = self._get_nearest_expiry()
        if not expiry_date:
            return None

        # Get base symbol for matching (e.g., NIFTY 50 -> NIFTY)
        raw_sym = self.index_symbol.split(':')[1]
        if "NIFTY 50" in raw_sym:
            sym_prefix = "NIFTY"
        elif "NIFTY BANK" in raw_sym:
            sym_prefix = "BANKNIFTY"
        else:
            sym_prefix = raw_sym.replace(" ", "")

        instruments = [inst for inst in self.all_instruments
                      if inst.instrument_type == option_type
                      and inst.expiry == expiry_date
                      and inst.symbol.startswith(sym_prefix)]

        if not instruments:
            # Try alternative symbol matching
            instruments = [inst for inst in self.all_instruments
                          if inst.instrument_type == option_type
                          and inst.expiry == expiry_date
                          and sym_prefix in inst.symbol]

        best_inst = None
        closest_delta_diff = float('inf')
        target_delta = (self.min_delta + self.max_delta) / 2

        today = datetime.datetime.now(self.tz).date()
        days_to_expiry = (expiry_date - today).days
        if days_to_expiry <= 0: days_to_expiry = 0.5 # Same day expiry

        for inst in instruments:
            # Calculate delta
            bs = mibian.BS([spot_price, inst.strike, self.interest_rate, days_to_expiry], volatility=self.todays_volatility)
            delta = abs(bs.callDelta if option_type == "CE" else bs.putDelta)

            if self.min_delta <= delta <= self.max_delta:
                diff = abs(delta - target_delta)
                if diff < closest_delta_diff:
                    closest_delta_diff = diff
                    best_inst = inst

        return best_inst

    def _get_nearest_expiry(self):
        """Finds nearest expiry for the index"""
        raw_sym = self.index_symbol.split(':')[1]
        if "NIFTY 50" in raw_sym:
            sym_prefix = "NIFTY"
        elif "NIFTY BANK" in raw_sym:
            sym_prefix = "BANKNIFTY"
        else:
            sym_prefix = raw_sym.replace(" ", "")

        expiries = sorted(list(set([inst.expiry for inst in self.all_instruments if sym_prefix in inst.symbol and inst.expiry])))
        today = datetime.datetime.now(self.tz).date()
        for exp in expiries:
            if exp >= today:
                return exp
        return None

    def calculate_quantity(self, premium: float, is_volatile: bool) -> int:
        """Calculates quantity based on fixed risk"""
        if premium <= 0: return self.lot_size

        # Risk per trade in currency
        risk_amt = self.capital * self.risk_per_trade
        # Stop loss in currency per unit
        sl_per_unit = premium * self.stop_loss_pct

        qty = int(risk_amt / sl_per_unit)
        # Round to lot size
        qty = (qty // self.lot_size) * self.lot_size

        if is_volatile:
            qty = int(qty * self.volatile_size_mult)
            qty = (qty // self.lot_size) * self.lot_size

        return max(qty, self.lot_size)

    def verify_position_closed(self, symbol: str) -> bool:
        """Verifies that the position for a given symbol is closed via broker.get_positions()"""
        try:
            positions = self.broker.get_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    return pos.quantity_total == 0
            return True
        except Exception as e:
            logger.error(f"Error verifying position: {e}")
            return False

    def execute_trade(self, state: str, spot_price: float, current_atr: float, vol_ok: bool = True):
        """Executes trade based on market state"""
        if "Choppy" in state:
            logger.info("Market is choppy. No new trades.")
            return

        if not vol_ok:
            logger.info("Volume confirmation failed. Skipping trade.")
            return

        option_type = "CE" if "Bullish" in state else "PE"
        is_volatile = "volatile" in state

        # Check if we already have the right position
        if self.current_position:
            if self.current_position['type'] == option_type:
                logger.info(f"Already holding {option_type}. No action.")
                return
            else:
                logger.info(f"Opposite position {self.current_position['type']} exists. Closing it.")
                prev_symbol = self.current_position['symbol']
                self.close_position()

                # Fill verification
                verified = False
                for _ in range(10): # Wait up to 10s
                    if self.verify_position_closed(prev_symbol):
                        verified = True
                        break
                    logger.info(f"Waiting for {prev_symbol} fill verification...")
                    time.sleep(1)

                if not verified:
                    logger.error(f"Could not verify closure of {prev_symbol}. Skipping entry.")
                    return

        # New Entry
        inst = self.select_strike(option_type, spot_price)
        if not inst:
            logger.error(f"Could not find suitable {option_type} strike.")
            return

        quote = self.broker.get_quote(f"{self.exchange}:{inst.symbol}")
        premium = quote.last_price

        # Liquidity check
        if quote.buy_quantity == 0 or quote.sell_quantity == 0:
            logger.warning(f"Low liquidity for {inst.symbol}. Skipping.")
            return

        qty = self.calculate_quantity(premium, is_volatile)

        logger.info(f"Opening {option_type} position: {inst.symbol} Qty: {qty} @ {premium}")

        req = OrderRequest(
            symbol=inst.symbol,
            exchange=Exchange.NFO,
            transaction_type=TransactionType.BUY,
            quantity=qty,
            product_type=self.product_type,
            order_type=OrderType.LIMIT,
            price=premium, # Use limit order near LTP
            tag=self.tag
        )

        resp = self.broker.place_order(req)
        if resp.order_id:
            # ATR-based stop: SL = entry_price ± TRAIL_ATR_MULT * ATR_intraday
            # Note: ATR is on index, but here we apply it to premium proportionally or as requested
            # Given the ambiguity, we'll use the tighter of fixed % or ATR-based.
            # For ATR-based on premium, we'd need option ATR or scale index ATR by delta.
            # Assuming Delta ~ 0.4, SL_dist = 0.4 * 1.5 * IndexATR
            # We'll use a simplified version: tighter of 20% or 2 * ATR (scaled by 0.4)
            approx_delta = 0.4
            atr_sl_dist = self.trail_atr_mult * current_atr * approx_delta
            fixed_sl_dist = premium * self.stop_loss_pct
            sl_dist = min(fixed_sl_dist, atr_sl_dist)

            self.current_position = {
                'symbol': inst.symbol,
                'type': option_type,
                'qty': qty,
                'entry_price': premium,
                'order_id': resp.order_id,
                'sl': premium - sl_dist,
                'highest_price': premium,
                'trailing': False
            }
            logger.info(f"Position opened. SL: {self.current_position['sl']}")

    def close_position(self, reason: str = "Signal flip"):
        """Closes the active position"""
        if not self.current_position:
            return

        logger.info(f"Closing position {self.current_position['symbol']} due to {reason}")

        req = OrderRequest(
            symbol=self.current_position['symbol'],
            exchange=Exchange.NFO,
            transaction_type=TransactionType.SELL,
            quantity=self.current_position['qty'],
            product_type=self.product_type,
            order_type=OrderType.MARKET,
            tag=self.tag
        )

        resp = self.broker.place_order(req)
        if resp.order_id:
            # Record P&L
            quote = self.broker.get_quote(f"{self.exchange}:{self.current_position['symbol']}")
            exit_price = quote.last_price
            pnl = (exit_price - self.current_position['entry_price']) * self.current_position['qty']
            self.daily_loss -= pnl # daily_loss tracks negative pnl
            logger.info(f"Position closed @ {exit_price}. P&L: {pnl}")
            self.current_position = None

    def manage_active_trade(self, spot_price: float, current_atr: float):
        """Manages active trade: SL, Trailing SL, Time Exit"""
        if not self.current_position:
            return

        # Check Time Exit
        now = datetime.datetime.now(self.tz).time()
        if now >= self.square_off_time:
            self.close_position(f"Square-off Time ({self.square_off_time})")
            return

        quote = self.broker.get_quote(f"{self.exchange}:{self.current_position['symbol']}")
        ltp = quote.last_price

        # Update highest price for trailing
        if ltp > self.current_position['highest_price']:
            self.current_position['highest_price'] = ltp

            # Start trailing if profit threshold hit
            gain_pct = (ltp - self.current_position['entry_price']) / self.current_position['entry_price']
            if gain_pct >= self.profit_threshold_pct:
                self.current_position['trailing'] = True
                logger.info("Profit threshold hit. Starting trailing SL.")

        # Update SL if trailing
        if self.current_position['trailing']:
            # Trail based on ATR scaled by delta
            approx_delta = 0.4
            new_sl = ltp - (self.trail_atr_mult * current_atr * approx_delta)
            if new_sl > self.current_position['sl']:
                self.current_position['sl'] = new_sl
                logger.debug(f"Updated trailing SL to {new_sl}")

        # Check SL
        if ltp <= self.current_position['sl']:
            self.close_position("Stop Loss Hit")
            return

    def run_iteration(self):
        """Main strategy iteration"""
        # 1. Check Daily Loss
        if self.daily_loss >= self.max_daily_loss:
            logger.warning("Max daily loss hit. Stopping for the day.")
            if self.current_position:
                self.close_position("Max Daily Loss")
            return

        # 2. Fetch Historical Data
        now_ist = datetime.datetime.now(self.tz)
        end_date = now_ist.strftime("%Y-%m-%d")
        start_date = (now_ist - datetime.timedelta(days=5)).strftime("%Y-%m-%d")

        try:
            candles = self.broker.get_history(self.index_symbol, "5m", start_date, end_date)
            if not candles:
                logger.error("Failed to fetch historical data")
                return

            df = pd.DataFrame(candles)
            df = self.calculate_indicators(df)
            state = self.get_market_state(df)
            last_candle = df.iloc[-1]
            spot_price = last_candle['close']
            current_atr = last_candle['atr']
            vol_ok = last_candle['vol_ok']

            logger.info(f"State: {state} | Spot: {spot_price} | ATR: {current_atr:.2f} | Vol OK: {vol_ok}")

            # 3. Handle Active Trade
            self.manage_active_trade(spot_price, current_atr)

            # 4. Check for New Entries
            now = datetime.datetime.now(self.tz).time()
            if self.start_trading_time <= now <= self.no_new_trades_after:
                self.execute_trade(state, spot_price, current_atr, vol_ok)
            elif now > self.last_exit_time:
                if self.current_position:
                    logger.info("Late session. Managing exits.")
                    # In late session we just hold or exit via manage_active_trade

        except Exception as e:
            logger.error(f"Error in strategy iteration: {e}", exc_info=True)

if __name__ == "__main__":
    # Standard boilerplate for running strategy
    import argparse
    from dispatcher import DataDispatcher
    from orders import OrderTracker
    from queue import Queue

    parser = argparse.ArgumentParser(description="Supertrend Strategy")
    parser.add_argument('--config', type=str, default="strategy/configs/supertrend.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['default']

    broker = BrokerGateway.from_name(os.getenv("BROKER_NAME"))
    order_tracker = OrderTracker()

    strategy = SupertrendStrategy(config, broker, order_tracker)

    logger.info("Starting Supertrend Strategy Loop")
    try:
        while True:
            strategy.run_iteration()
            time.sleep(60) # Run every minute
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
