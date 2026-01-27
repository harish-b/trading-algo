import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
from types import SimpleNamespace
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

    This strategy uses the Supertrend indicator on an underlying index (e.g., NIFTY 50)
    to identify trends and executes directional options trades (buying CE or PE).
    It incorporates market regime detection based on ATR to adjust position sizing
    and handles complex execution rules like fill verification and time-based exits.
    """

    def __init__(self, config: Dict, broker: BrokerGateway, order_tracker=None):
        """
        Initializes the strategy with configuration, broker gateway, and optional order tracker.

        Args:
            config (Dict): Strategy parameters from YAML.
            broker (BrokerGateway): Facade for broker operations.
            order_tracker: Optional utility for tracking order status.
        """
        self.config = config
        self.broker = broker
        self.order_tracker = order_tracker

        # --- Core Parameters ---
        self.index_symbol = config.get("index_symbol", "NSE:NIFTY 50")
        self.exchange = config.get("exchange", "NFO")

        # --- Indicator Parameters ---
        self.st_period = config.get("st_period", 10)
        self.st_multiplier = config.get("st_multiplier", 3.0)
        self.atr_period = config.get("atr_period", 14)

        # --- Risk Management Parameters ---
        self.capital = config.get("capital", 1000000)
        self.risk_per_trade = config.get("risk_per_trade", 0.01) # 1% of capital
        self.stop_loss_pct = config.get("stop_loss_pct", 0.20)  # 20% of premium
        self.trail_atr_mult = config.get("trail_atr_mult", 1.5) # ATR multiplier for trailing
        self.profit_threshold_pct = config.get("profit_threshold_pct", 0.30) # Start trailing after 30% gain

        # --- Order & Product Settings ---
        self.product_type = ProductType.MARGIN if config.get("product_type", "NRML") == "NRML" else ProductType.INTRADAY
        self.tag = config.get("tag", "SUPERTREND_STRAT")
        self.lot_size = config.get("lot_size", 65)

        # --- Strike Selection & Regime Parameters ---
        self.min_delta = config.get("min_delta", 0.30)
        self.max_delta = config.get("max_delta", 0.45)
        self.volatile_size_mult = config.get("volatile_size_mult", 0.75) # Reduce size in volatile regimes

        # --- Timezone handling (Indian Market uses IST) ---
        self.tz = ZoneInfo("Asia/Kolkata")

        # --- Trading Session Windows (parsed as IST) ---
        self.start_trading_time = datetime.datetime.strptime(config.get("start_trading_time", "09:40"), "%H:%M").time()
        self.no_new_trades_after = datetime.datetime.strptime(config.get("no_new_trades_after", "14:30"), "%H:%M").time()
        self.last_exit_time = datetime.datetime.strptime(config.get("last_exit_time", "15:00"), "%H:%M").time()
        self.square_off_time = datetime.datetime.strptime(config.get("square_off_time", "15:15"), "%H:%M").time()

        # --- Filters ---
        self.min_time_gap_candles = config.get("min_time_gap_candles", 3) # Anti-whipsaw filter
        self.require_volume_confirmation = config.get("require_volume_confirmation", True)
        self.volume_period = config.get("volume_period", 20)

        # --- Internal State Trackers ---
        self.current_position: Optional[Dict] = None # Stores active trade details (symbol, entry_price, sl, etc.)
        self.last_flip_index = -1 # Index of the last Supertrend direction change
        self.last_signal = None # 'BUY' (Uptrend) or 'SELL' (Downtrend)
        self.daily_loss = 0 # Cumulative P&L tracker for the day
        self.last_trading_day = datetime.datetime.now(self.tz).date() # For daily loss reset
        self.max_daily_loss = self.capital * 0.02 # Hard stop at 2% daily loss

        # Initialize broker and download instrument master
        logger.info("Initializing Supertrend Strategy...")
        self.broker.download_instruments()
        self.all_instruments = self.broker.get_instruments()

        # Parameters for Options Greeks (Mibian)
        self.interest_rate = config.get("interest_rate", 10.0)
        self.todays_volatility = config.get("todays_volatility", 20.0)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates ATR (Average True Range) and Supertrend indicators on the provided DataFrame.
        Also computes auxiliary flags like ATR direction and volume confirmation.
        """
        # 1. Calculate True Range (TR) and ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['tr'] = np.max(ranges, axis=1)
        df['atr'] = df['tr'].rolling(self.atr_period).mean()

        # 2. Calculate Supertrend
        # Formula: (High + Low) / 2 +/- Multiplier * ATR
        hl2 = (df['high'] + df['low']) / 2
        df['upperband'] = hl2 + (self.st_multiplier * df['atr'])
        df['lowerband'] = hl2 - (self.st_multiplier * df['atr'])
        df['is_uptrend'] = True # Default state

        # Iteratively calculate Supertrend to handle band logic (non-repainting)
        for i in range(1, len(df)):
            if pd.isna(df['upperband'].iloc[i-1]) or pd.isna(df['lowerband'].iloc[i-1]):
                continue

            # Trend Flip Logic
            if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
                df.at[df.index[i], 'is_uptrend'] = True
            elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
                df.at[df.index[i], 'is_uptrend'] = False
            else:
                # Maintain previous trend
                df.at[df.index[i], 'is_uptrend'] = df['is_uptrend'].iloc[i-1]

                # Band tightening logic: ensures Supertrend line only moves closer to price
                if df['is_uptrend'].iloc[i] and df['lowerband'].iloc[i] < df['lowerband'].iloc[i-1]:
                    df.at[df.index[i], 'lowerband'] = df['lowerband'].iloc[i-1]
                if not df['is_uptrend'].iloc[i] and df['upperband'].iloc[i] > df['upperband'].iloc[i-1]:
                    df.at[df.index[i], 'upperband'] = df['upperband'].iloc[i-1]

        # Final Supertrend Line
        df['st'] = np.where(df['is_uptrend'], df['lowerband'], df['upperband'])

        # 3. Detect Market Regimes
        df['atr_rising'] = df['atr'] > df['atr'].shift(1) # Indicates rising volatility

        # 4. Volume confirmation (Current Volume > SMA of Volume)
        if self.require_volume_confirmation:
            df['vol_avg'] = df['volume'].rolling(self.volume_period).mean()
            df['vol_ok'] = df['volume'] > df['vol_avg']
        else:
            df['vol_ok'] = True

        return df

    def get_market_state(self, df: pd.DataFrame) -> str:
        """
        Classifies the market state based on the latest indicators.
        States: Bullish calm, Bullish volatile, Bearish calm, Bearish volatile, Choppy.
        """
        last_candle = df.iloc[-1]
        is_uptrend = last_candle['is_uptrend']
        atr_rising = last_candle['atr_rising']
        current_index = len(df) - 1

        # 1. Detect Directional Flip and Confirmation
        if len(df) > 1:
            prev_candle = df.iloc[-2]
            if prev_candle['is_uptrend'] != is_uptrend:
                # Signal changed (flip)
                self.last_flip_index = current_index

                # Confirmation Filter: Last candle close must be beyond the Supertrend line
                if is_uptrend and last_candle['close'] <= last_candle['st']:
                    return "Choppy" # False signal
                if not is_uptrend and last_candle['close'] >= last_candle['st']:
                    return "Choppy" # False signal

        # 2. Anti-whipsaw filter: Ignore signals that happen too frequently
        if self.last_flip_index != -1 and (current_index - self.last_flip_index) < self.min_time_gap_candles:
            return "Choppy"

        # 3. Market Regime Classification
        if is_uptrend:
            return "Bullish volatile" if atr_rising else "Bullish calm"
        else:
            return "Bearish volatile" if atr_rising else "Bearish calm"

    def select_strike(self, option_type: str, spot_price: float) -> Optional[Dict]:
        """
        Selects a near-ATM option strike based on a target Delta range (0.30 - 0.45).
        Uses Mibian for Black-Scholes Greeks calculation.
        """
        expiry_date = config.get("nearest_expiry", "2026-02-03")
        if not expiry_date:
            return None

        sym_prefix = config.get("sym_prefix", "NIFTY26203")

        # Normalize expiry_date to a date object if necessary
        if isinstance(expiry_date, str):
            try:
                expiry_date = datetime.datetime.strptime(expiry_date, "%Y-%m-%d").date()
            except Exception:
                # leave as-is if parsing fails
                pass
        elif isinstance(expiry_date, datetime.datetime):
            expiry_date = expiry_date.date()

        # Filter instruments by type, expiry and symbol prefix
        

        df = self.all_instruments

        # Ensure expiry is datetime
        df['expiry'] = pd.to_datetime(df['expiry'])
        expiry_date = pd.to_datetime(expiry_date)

        mask = (
            (df['instrument_type'] == option_type) &
            (df['expiry'] == expiry_date) &
            (df['symbol'].str.contains(sym_prefix, na=False))   # more reliable than startswith
        )

        instruments = [SimpleNamespace(**row.to_dict()) for _, row in df.loc[mask].iterrows()]

        print(f"Found {len(instruments)} instruments")


        print(f"Found {len(instruments)} instruments for {option_type} with prefix {sym_prefix} expiring on {expiry_date}")
        print([inst.symbol for inst in instruments])
        best_inst = None
        closest_delta_diff = float('inf')
        target_delta = (self.min_delta + self.max_delta) / 2 # Mid-point of range

        today = datetime.datetime.now(self.tz).date()
        days_to_expiry = (expiry_date.date() - today).days
        if days_to_expiry <= 0: days_to_expiry = 0.5 # Same day expiry handling

        # Iterate and find the strike closest to the target Delta
        for inst in instruments:
            # Calculate delta via Black-Scholes
            bs = mibian.BS([spot_price, inst.strike, self.interest_rate, days_to_expiry], volatility=self.todays_volatility)
            delta = abs(bs.callDelta if option_type == "CE" else bs.putDelta)
            print(f"Strike: {inst.strike} | Delta: {delta:.4f}")
            # Check if within acceptable range
            if self.min_delta <= delta <= self.max_delta:
                diff = abs(delta - target_delta)
                if diff < closest_delta_diff:
                    closest_delta_diff = diff
                    best_inst = inst

        return best_inst

    def calculate_quantity(self, premium: float, is_volatile: bool) -> int:
        """
        Calculates position size based on a fixed risk model (e.g., risk 1% of capital per trade).
        Scales down quantity in volatile regimes.
        """
        if premium <= 0: return self.lot_size

        # Risk amount = Capital * % Risk
        risk_amt = self.capital * self.risk_per_trade
        # Stop loss amount per unit (based on premium)
        sl_per_unit = premium * self.stop_loss_pct

        # Base Quantity
        qty = int(risk_amt / sl_per_unit)
        # Round to lot size
        qty = (qty // self.lot_size) * self.lot_size

        # Adjust for Volatile Regime
        if is_volatile:
            qty = int(qty * self.volatile_size_mult)
            qty = (qty // self.lot_size) * self.lot_size

        return max(qty, self.lot_size) # Minimum 1 lot

    def verify_position_closed(self, symbol: str) -> bool:
        """Safety check: ensures the position is fully closed by polling the broker."""
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
        """
        Orchestrates trade entries and trend-flip exits.
        Maintains the invariant: at most one active long option trade.
        """
        if "Choppy" in state:
            logger.info("Market is choppy. No new trades.")
            return

        # if not vol_ok:
        #     logger.info("Volume confirmation failed. Skipping trade.")
        #     return

        option_type = "CE" if "Bullish" in state else "PE"
        is_volatile = "volatile" in state

        # 1. Manage Trend Flip
        if self.current_position:
            if self.current_position['type'] == option_type:
                logger.info(f"Already holding {option_type}. No action.")
                return
            else:
                # Opposite trend detected - Close existing position first
                logger.info(f"Opposite position {self.current_position['type']} exists. Closing it.")
                prev_symbol = self.current_position['symbol']
                self.close_position()

                # Fill verification: wait for closure to reflect in positions()
                verified = False
                for _ in range(10): # Max 10 attempts
                    if self.verify_position_closed(prev_symbol):
                        verified = True
                        break
                    logger.info(f"Waiting for {prev_symbol} fill verification...")
                    time.sleep(1)

                if not verified:
                    logger.error(f"Could not verify closure of {prev_symbol}. Skipping entry to avoid hedge.")
                    return

        # 2. New Entry Logic
        inst = self.select_strike(option_type, spot_price)
        if not inst:
            logger.error(f"Could not find suitable {option_type} strike.")
            return

        quote = self.broker.get_quote(f"{self.exchange}:{inst.symbol}")
        premium = quote.last_price

        print(f"Selected {inst.symbol} at premium {premium} and quote {quote}")

        # Liquidity guard
        # if quote.buy_quantity == 0 or quote.sell_quantity == 0:
        #     logger.warning(f"Low liquidity for {inst.symbol}. Skipping.")
        #     return

        qty = self.calculate_quantity(premium, is_volatile)

        logger.info(f"Opening {option_type} position: {inst.symbol} Qty: {qty} @ {premium}")

        # Place Limit Order at current market premium to control slippage
        req = OrderRequest(
            symbol=inst.symbol,
            exchange=Exchange.NFO,
            transaction_type=TransactionType.BUY,
            quantity=qty,
            product_type=self.product_type,
            order_type=OrderType.LIMIT,
            price=premium,
            tag=self.tag
        )

        resp = self.broker.place_order(req)
        if resp.order_id:
            # Initialize SL as the tighter of: Fixed % of premium or ATR-based distance
            approx_delta = 0.4 # Directional options usually move at ~0.4 delta
            atr_sl_dist = self.trail_atr_mult * current_atr * approx_delta
            fixed_sl_dist = premium * self.stop_loss_pct
            sl_dist = min(fixed_sl_dist, atr_sl_dist)

            # Record internal state for the trade
            self.current_position = {
                'symbol': inst.symbol,
                'type': option_type,
                'qty': qty,
                'entry_price': premium,
                'order_id': resp.order_id,
                'sl': premium - sl_dist, # Initial Stop Loss
                'highest_price': premium, # Tracker for Trailing Stop
                'trailing': False # Flip to True when profit threshold hit
            }
            logger.info(f"Position opened. SL: {self.current_position['sl']}")

    def close_position(self, reason: str = "Signal flip"):
        """Executes a quick Market exit for the current active position."""
        if not self.current_position:
            return

        logger.info(f"Closing position {self.current_position['symbol']} due to {reason}")

        req = OrderRequest(
            symbol=self.current_position['symbol'],
            exchange=Exchange.NFO,
            transaction_type=TransactionType.SELL,
            quantity=self.current_position['qty'],
            product_type=self.product_type,
            order_type=OrderType.MARKET, # Immediate exit
            tag=self.tag
        )

        resp = self.broker.place_order(req)
        if resp.order_id:
            # Post-exit accounting
            quote = self.broker.get_quote(f"{self.exchange}:{self.current_position['symbol']}")
            exit_price = quote.last_price
            pnl = (exit_price - self.current_position['entry_price']) * self.current_position['qty']
            self.daily_loss -= pnl # TRACKER: Tracks loss as positive values
            logger.info(f"Position closed @ {exit_price}. P&L: {pnl}")
            self.current_position = None

    def manage_active_trade(self, spot_price: float, current_atr: float):
        """
        Ongoing management of active trades: Stop Loss, Trailing, and Mandatory Time Exits.
        """
        if not self.current_position:
            return

        # 1. Mandatory Square-off Exit
        now = datetime.datetime.now(self.tz).time()
        if now >= self.square_off_time:
            self.close_position(f"Square-off Time ({self.square_off_time})")
            return

        # Fetch current premium
        quote = self.broker.get_quote(f"{self.exchange}:{self.current_position['symbol']}")
        ltp = quote.last_price

        # 2. Update Highest Price for Trailing Stop
        if ltp > self.current_position['highest_price']:
            self.current_position['highest_price'] = ltp

            # Start trailing only after a significant gain (e.g. 30%)
            gain_pct = (ltp - self.current_position['entry_price']) / self.current_position['entry_price']
            if gain_pct >= self.profit_threshold_pct:
                if not self.current_position['trailing']:
                    self.current_position['trailing'] = True
                    logger.info("Profit threshold hit. Starting trailing SL.")

        # 3. Dynamic Trailing SL Logic
        if self.current_position['trailing']:
            # Trail based on ATR scaled by option delta
            approx_delta = 0.4
            new_sl = ltp - (self.trail_atr_mult * current_atr * approx_delta)

            # Only move SL UP (tighten)
            if new_sl > self.current_position['sl']:
                self.current_position['sl'] = new_sl
                logger.debug(f"Updated trailing SL to {new_sl}")

        # 4. Check Exit Conditions (Fixed or Trailing SL)
        if ltp <= self.current_position['sl']:
            self.close_position("Stop Loss Hit")
            return

    def run_iteration(self):
        """
        Single main strategy heartbeat.
        Checks daily limits, calculates signals, and manages trades.
        """
        # 1. Reset Daily P&L Tracker at start of new day
        now_ist = datetime.datetime.now(self.tz)
        current_date = now_ist.date()
        if current_date > self.last_trading_day:
            logger.info(f"New trading day detected: {current_date}. Resetting daily P&L.")
            self.daily_loss = 0
            self.last_trading_day = current_date

        # 2. Daily Risk Check
        if self.daily_loss >= self.max_daily_loss:
            logger.warning("Max daily loss hit (2%). Stopping trading for the day.")
            if self.current_position:
                self.close_position("Max Daily Loss Safeguard")
            return

        # 3. Fetch and Process Market Data
        end_date = now_ist.strftime("%Y-%m-%d")
        start_date = (now_ist - datetime.timedelta(days=5)).strftime("%Y-%m-%d")

        try:
            # Pull 1m candles for signal calculation
            print(self.index_symbol, start_date, end_date)
            candles = self.broker.get_history(self.index_symbol, "1m", start_date, end_date)
            if not candles:
                logger.error("Failed to fetch historical data")
                return

            print(f"Fetched {len(candles)} candles")
            print(candles[-5:])
            df = pd.DataFrame(candles)
            df = self.calculate_indicators(df)
            state = self.get_market_state(df)

            last_candle = df.iloc[-1]
            spot_price = last_candle['close']
            current_atr = last_candle['atr']
            vol_ok = last_candle['vol_ok']

            logger.info(f"State: {state} | Spot: {spot_price} | ATR: {current_atr:.2f} | Vol OK: {vol_ok}")

            # 4. Manage Open Trades (SL/Trailing)
            self.manage_active_trade(spot_price, current_atr)

            # 5. Evaluate Signal-based Entries
            now = now_ist.time()
            # Entry restricted to specific window
            if self.start_trading_time <= now <= self.no_new_trades_after:
                self.execute_trade(state, spot_price, current_atr, vol_ok)
            elif now > self.last_exit_time:
                # Late session: Manage existing exits but don't enter new trades
                if self.current_position:
                    logger.info("Late session. Managing exits only.")

        except Exception as e:
            logger.error(f"Error in strategy iteration: {e}", exc_info=True)

if __name__ == "__main__":
    # --- CLI Entry Point ---
    import argparse
    from dispatcher import DataDispatcher
    from orders import OrderTracker
    from queue import Queue

    parser = argparse.ArgumentParser(description="Supertrend Strategy Runner")
    parser.add_argument('--config', type=str, default="configs/supertrend.yml",
                        help="Path to YAML configuration")
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['default']

    # Initialize Components
    broker = BrokerGateway.from_name(os.getenv("BROKER_NAME"))
    order_tracker = OrderTracker()

    # Instantiate Strategy
    strategy = SupertrendStrategy(config, broker, order_tracker)

    logger.info("Starting Supertrend Strategy Monitoring Loop (1-minute intervals)")
    try:
        while True:
            strategy.run_iteration()
            time.sleep(15) # Heartbeat interval
    except KeyboardInterrupt:
        logger.info("Strategy loop terminated by user.")
