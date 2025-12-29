"""ロギング設定"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    ロガーを取得

    Args:
        name: ロガー名
        log_file: ログファイルパス（Noneの場合はファイル出力なし）
        level: ログレベル

    Returns:
        Logger: 設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # すでにハンドラが設定されている場合はスキップ
    if logger.handlers:
        return logger

    # フォーマット設定
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # コンソール出力
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイル出力
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
