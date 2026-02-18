from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "yfinance is required. Install it with: pip install yfinance"
    ) from e


@dataclass(frozen=True)
class PriceDataResult:
    prices: pd.DataFrame  # columns: tickers, index: dates
    source: str           # "download" or "cache"
    cache_path: Path | None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fetch_price_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str | None = None,
    price_field: str = "Adj Close",
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> PriceDataResult:
    """
    Fetches historical price data for tickers. Optionally caches to CSV.

    Returns a DataFrame with:
      - index = DatetimeIndex of trading days
      - columns = tickers
      - values = prices (float)
    """
    tickers = tuple(dict.fromkeys([t.strip().upper() for t in tickers if str(t).strip()]))
    if not tickers:
        raise ValueError("No tickers provided.")

    cache_path: Path | None = None
    if cache_dir is not None:
        _ensure_dir(cache_dir)
        end_tag = end_date if end_date is not None else "latest"
        cache_path = cache_dir / f"prices_{'-'.join(tickers)}_{start_date}_to_{end_tag}_{price_field.replace(' ', '')}.csv"

        if use_cache and cache_path.exists():
            prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            # Ensure ordering of columns matches tickers
            prices = prices[[c for c in tickers if c in prices.columns]]
            return PriceDataResult(prices=prices, source="cache", cache_path=cache_path)

    # Download from yfinance
    df = yf.download(list(tickers), start=start_date, end=end_date, auto_adjust=False, progress=False)

    if df.empty:
        raise RuntimeError("yfinance returned no data. Check tickers or date range.")

    # yf returns MultiIndex columns when multiple tickers
    # df columns might look like ('Adj Close','AAPL') or ('Close','AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        if price_field not in df.columns.get_level_values(0):
            raise KeyError(f"'{price_field}' not found in downloaded data. Available: {sorted(set(df.columns.get_level_values(0)))}")
        prices = df[price_field].copy()
    else:
        # Single ticker case
        if price_field not in df.columns:
            raise KeyError(f"'{price_field}' not found in downloaded data. Available: {list(df.columns)}")
        prices = df[[price_field]].copy()
        # rename column to ticker for consistency
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all")
    prices = prices.sort_index()

    # Cache
    if cache_path is not None:
        prices.to_csv(cache_path)

    return PriceDataResult(prices=prices, source="download", cache_path=cache_path)
