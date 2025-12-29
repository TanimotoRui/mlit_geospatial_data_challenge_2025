"""評価指標"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE (Root Mean Squared Error)
    
    Args:
        y_true: 真値
        y_pred: 予測値
    
    Returns:
        float: RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSLE (Root Mean Squared Logarithmic Error)
    
    Args:
        y_true: 真値
        y_pred: 予測値
    
    Returns:
        float: RMSLE
    """
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE (Mean Absolute Percentage Error)
    
    Args:
        y_true: 真値
        y_pred: 予測値
    
    Returns:
        float: MAPE (%)
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    各種評価指標を計算
    
    Args:
        y_true: 真値
        y_pred: 予測値
    
    Returns:
        dict: 評価指標の辞書
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

