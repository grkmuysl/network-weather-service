"""Kurulumun doğru yapıldığını kontrol eden testler."""

import importlib

import pytest


@pytest.mark.parametrize(
    "package",
    ["numpy", "pandas", "statsmodels", "torch", "matplotlib"],
)
def test_package_imports(package):
    importlib.import_module(package)


def test_nwl_package_imports():
    nwl = importlib.import_module("nwl")
    assert nwl.__version__


def test_arima_runs():
    """statsmodels gerçekten çalışıyor mu?"""
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    rng = np.random.default_rng(0)
    series = np.sin(np.linspace(0, 12, 100)) + rng.normal(0, 0.1, 100)
    forecast = ARIMA(series, order=(1, 0, 1)).fit().forecast(steps=3)
    assert len(forecast) == 3


def test_lstm_runs():
    """PyTorch ile küçük bir LSTM gerçekten çalışıyor mu?"""
    import torch

    lstm = torch.nn.LSTM(input_size=1, hidden_size=8, batch_first=True)
    output, _ = lstm(torch.zeros(2, 10, 1))
    assert output.shape == (2, 10, 8)
