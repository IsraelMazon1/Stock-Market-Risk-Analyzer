from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Data settings
    tickers: tuple[str, ...] = ("ADBE", "NVDA", "AMZN", "GOOG", "SOFI")
    start_date: str = "2020-01-01"
    end_date: str | None = None  # None => up to latest
    price_field: str = "Adj Close"  # yfinance uses "Adj Close"

    # Finance settings
    trading_days: int = 252
    confidence_level: float = 0.95
    risk_free_rate_annual: float = 0.0  # set to e.g. 0.04 if you want

    # Project paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"

DEFAULT_CONFIG = Config()
