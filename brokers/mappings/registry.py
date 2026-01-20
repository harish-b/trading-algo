from __future__ import annotations

from typing import Dict

from ..core.enums import OrderType, ProductType, TransactionType, Validity


class MappingRegistry:
    """Holds per-broker mapping tables for enums/strings."""

    order_type: Dict[str, Dict[OrderType, str | int]] = {}
    product_type: Dict[str, Dict[ProductType, str]] = {}
    transaction_type: Dict[str, Dict[TransactionType, str | int]] = {}
    validity: Dict[str, Dict[Validity, str]] = {}

    @classmethod
    def register_default(cls) -> None:
        # Zerodha
        cls.order_type["zerodha"] = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "SL-M",
            OrderType.STOP_LIMIT: "SL",
        }
        cls.product_type["zerodha"] = {
            ProductType.INTRADAY: "MIS",
            ProductType.CNC: "CNC",
            ProductType.MARGIN: "NRML",
        }
        cls.transaction_type["zerodha"] = {
            TransactionType.BUY: "BUY",
            TransactionType.SELL: "SELL",
        }
        cls.validity["zerodha"] = {Validity.DAY: "DAY", Validity.IOC: "IOC"}


# Initialize defaults
MappingRegistry.register_default()


