"""Data analysis utilities with some structural duplication — common in data science."""
import numpy as np


def summarize_train_data(data, feature_cols: list) -> dict:
    """Summarize training dataset statistics."""
    stats = {}
    for col in feature_cols:
        col_data = data[col].dropna()
        stats[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "missing": int(data[col].isna().sum()),
        }
    return stats


def summarize_test_data(data, feature_cols: list) -> dict:
    """Summarize test dataset statistics."""
    stats = {}
    for col in feature_cols:
        col_data = data[col].dropna()
        stats[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "missing": int(data[col].isna().sum()),
        }
    return stats


def summarize_validation_data(data, feature_cols: list) -> dict:
    """Summarize validation dataset statistics."""
    stats = {}
    for col in feature_cols:
        col_data = data[col].dropna()
        stats[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "missing": int(data[col].isna().sum()),
        }
    return stats
