from __future__ import annotations

from .registry import symbol_registry
from ..core.enums import Exchange


def _zerodha_resolver(internal: str) -> str:
    if ":" not in internal:
        internal = f"{Exchange.NSE.value}:{internal}"
    exch, sym = internal.split(":", 1)
    sym_u = sym.upper()
    index_mapping = {
        "NIFTY50-INDEX": "NIFTY 50",
        "NIFTYBANK-INDEX": "NIFTY BANK",
        "FINNIFTY-INDEX": "FINNIFTY",
    }
    if sym_u in index_mapping:
        return f"{exch}:{index_mapping[sym_u]}"
    if sym_u.endswith("-EQ"):
        sym = sym[:-3]
    return f"{exch}:{sym}"


# Register default resolvers
symbol_registry.register_resolver("zerodha", _zerodha_resolver)


