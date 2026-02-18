import pandas as pd
from src.risk_metrics import var_historical


def test_var_historical():
    returns = pd.Series([-0.02, 0.01, -0.03, 0.02])
    var = var_historical(returns, 0.95)
    assert var >= 0
