from enum import Enum


class ModelType(str, Enum):
    """
    Supported model types.
    Used for CLI typing i.e. see predict.py
    """
    cba_pure = "cba_pure"
    cba_stats = "cba_stats"
    xgb = "xgb"
    gbcxgb = "gbcxgb"
    csaxgb = "csaxgb"
    attention = "attention"
    mbcxgb = "mbcxgb"
