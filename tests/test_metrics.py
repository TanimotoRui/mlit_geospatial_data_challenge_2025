"""評価指標のテスト"""

import numpy as np
import pytest

from src.utils.metrics import rmse, rmsle, mape, calculate_metrics


def test_rmse():
    """RMSEのテスト"""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    expected = np.sqrt(((3-2.5)**2 + (-0.5-0)**2 + (2-2)**2 + (7-8)**2) / 4)
    assert np.isclose(rmse(y_true, y_pred), expected)


def test_rmsle():
    """RMSLEのテスト"""
    y_true = np.array([3, 5, 2.5, 7])
    y_pred = np.array([2.5, 5, 4, 8])
    
    result = rmsle(y_true, y_pred)
    assert result >= 0  # RMSLEは常に非負


def test_mape():
    """MAPEのテスト"""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    
    expected = np.mean([10/100, 10/200, 10/300]) * 100
    assert np.isclose(mape(y_true, y_pred), expected)


def test_calculate_metrics():
    """全評価指標の計算テスト"""
    y_true = np.array([3, 5, 2.5, 7])
    y_pred = np.array([2.5, 5, 4, 8])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "rmsle" in metrics
    assert "mape" in metrics
    assert "r2" in metrics
    
    # すべての値が数値であることを確認
    for value in metrics.values():
        assert isinstance(value, (int, float))

