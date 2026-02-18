import pandas as pd
from src.returns import compute_log_returns


def test_compute_log_returns():
    data = pd.DataFrame({
        "A": [100, 101, 102]
    })
    rets = compute_log_returns(data)
    assert not rets.empty
    assert len(rets) == 2
