"""Feature engineering shared by training and serving, so the two can't drift apart."""

from collections import deque
from threading import Lock

import pandas as pd

ROLLING_WINDOW = 5
RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALED_COLS = ["Amount", "Time", "Time_Diff", "Amount_Rolling_Avg_5"]


def add_stream_features(df: pd.DataFrame) -> pd.DataFrame:
    """Batch version, used in training. df must already be sorted by Time."""
    out = df.copy()
    out["Time_Diff"] = out["Time"].diff().fillna(0)
    out["Amount_Rolling_Avg_5"] = (
        out["Amount"].rolling(window=ROLLING_WINDOW, min_periods=1).mean())
    return out


class StreamState:
    """
    Streaming version, used by the API. Keeps the last few transactions so each
    new one gets the same Time_Diff and rolling average it would have had in
    training. Like the training data, this is one global stream (the Kaggle
    dataset has no card ID), so the state is shared across all requests.
    """

    def __init__(self):
        self._lock = Lock()
        self._last_time = None
        self._amounts = deque(maxlen=ROLLING_WINDOW)

    def reset(self):
        with self._lock:
            self._last_time = None
            self._amounts.clear()

    def update(self, time: float, amount: float) -> dict:
        with self._lock:
            time_diff = 0.0 if self._last_time is None else time - self._last_time
            self._last_time = time
            self._amounts.append(amount)
            rolling = sum(self._amounts) / len(self._amounts)
        return {"Time_Diff": time_diff, "Amount_Rolling_Avg_5": rolling}
