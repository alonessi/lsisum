from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ru_liquidity_sentinel.nsvm_model import NSVMAnomalyDetector
from ru_liquidity_sentinel.nsvm_lsi import ECDFCalibration


def test_context_sequences_use_previous_days_without_zero_padding():
    model = NSVMAnomalyDetector(seq_len=3)
    context = pd.DataFrame([[1.0], [2.0], [3.0]], columns=["x"])
    new = pd.DataFrame([[4.0], [5.0]], columns=["x"])

    seq = model._create_sequences(new, context=context).numpy()

    assert seq.shape == (2, 3, 1)
    assert seq[0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert seq[1, :, 0].tolist() == [2.0, 3.0, 4.0]


def test_save_load_weights_preserves_predictions(tmp_path):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=40, freq="D")
    X = pd.DataFrame(rng.normal(size=(40, 3)), index=idx, columns=["a", "b", "c"])

    model = NSVMAnomalyDetector(seq_len=4, hidden_dim=8, epochs=1, batch_size=8)
    model.fit(X)
    before = model.predict(X.tail(5), context=X.iloc[:-5])

    path = tmp_path / "nsvm_weights.pt"
    model.save_weights(path, feature_cols=list(X.columns), extra={"marker": "ok"})
    loaded = NSVMAnomalyDetector.load_weights(path)
    after = loaded.predict(X.tail(5), context=X.iloc[:-5])

    assert loaded.feature_cols_ == list(X.columns)
    assert loaded.extra_["marker"] == "ok"
    assert np.allclose(before, after)


def test_ecdf_calibration_returns_percentile_rank():
    train = np.arange(1000, dtype=float)
    cal = ECDFCalibration.fit(train)

    out = cal.transform(np.array([0.0, 250.0, 500.0, 800.0, 995.0, 1200.0]))

    assert np.all(np.diff(out) >= 0.0)
    assert np.allclose(out[1:5], [25.05, 50.05, 80.05, 99.55])
    assert out[5] == 100.0
