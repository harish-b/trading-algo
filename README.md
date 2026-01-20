# trading-algo
Code for certain trading strategies
1. [Survivor Algo](https://raahibhushan.substack.com/p/the-survivor-an-algorithm-for-when)
2. [Wave Algo](https://raahibhushan.substack.com/p/the-wave-extractor-algo-2)

### Note
Please test the code on your own setup and verify that it works as expected.  
If you encounter any issues or unexpected behavior, feel free to [open an issue](../../issues) with relevant details so it can be addressed promptly.

## Disclaimer:
This algorithm is provided for **educational** and **informational purposes** only. Trading in financial markets involves substantial risk, and you may lose all or more than your initial investment. By using this algorithm, you acknowledge that all trading decisions are made at your own risk and discretion. The creators of this algorithm assume no liability or responsibility for any financial losses or damages incurred through its use. **Always do your own research and consult with a qualified financial advisor before trading.**


## Setup

### 1. Install Dependencies

To insall uv, use:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or


```bash
pip install uv
```

This uses `uv` for dependency management. Install dependencies:
```bash
uv sync
```

Or if you prefer using pip:

```bash
pip install -r requirements.txt  # You may need to generate this from pyproject.toml
```

### 2. Environment Configuration

1. Copy the sample environment file:
   ```bash
   cp .sample.env .env
   ```

2. Edit `.env` and fill in your broker credentials:
   ```bash
    # should be - zerodha
    BROKER_NAME=<INPUT_YOUR_BROKER_NAME>
    BROKER_API_KEY=<INPUT_YOUR_API_KEY>
    BROKER_API_SECRET=<INPUT_YOUR_API_SECRET>
    BROKER_LOGIN_MODE='auto' # manual or auto
    BROKER_ID=<INPUT_YOUR_BROKER_ID>
    BROKER_TOTP_REDIDRECT_URI=<INPUT_YOUR_TOTP_REDIRECT_URI>
    BROKER_TOTP_KEY=<INPUT_YOUR_TOTP_KEY>
    BROKER_PASSWORD=<INPUT_YOUR_BROKER_PASSWORD>
   ```

### 3. Running Strategies

Strategies should be placed in the `strategy/` folder.

#### Running the Survivor Strategy


**Basic usage (using default config):**
```bash
cd strategy/
python survivor.py
```

**With custom parameters:**
```bash
cd strategy/
python survivor.py \
    --symbol-initials NIFTY25JAN30 \
    --pe-gap 25 --ce-gap 25 \
    --pe-quantity 50 --ce-quantity 50 \
    --min-price-to-sell 15
```

**View current configuration:**
```bash
cd strategy/
python survivor.py --show-config
```

### 4. Available Brokers

- **Zerodha**: Supports KiteConnect API with order management and live data streaming

### 5. Core Components

- `brokers/`: Broker implementations (Zerodha)
- `dispatcher.py`: Data routing and queue management - WIP
- `orders.py`: Order management utilities - WIP
- `logger.py`: Logging configuration
- `strategy/`: Place your trading strategies here

### Example Usage

```python
from brokers import BrokerGateway
broker = BrokerGateway.from_name(os.getenv("BROKER_NAME")) # zerodha
# Get historical data, place orders, etc.
```

For more details, check the individual broker implementations and example strategies in the `strategy/` folder.
