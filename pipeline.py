"""ML pipeline with intentionally complex functions — typical data science patterns."""
import numpy as np


def train_model(
    X_train,
    y_train,
    learning_rate: float,
    n_epochs: int,
    batch_size: int,
    regularization: float,
    dropout_rate: float,
    optimizer: str = "adam",
) -> dict:
    """Train a simple model. Many params is normal for ML training functions."""
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    
    for epoch in range(n_epochs):
        if optimizer == "adam":
            grad_scale = learning_rate * (1 - dropout_rate)
        elif optimizer == "sgd":
            grad_scale = learning_rate
        elif optimizer == "rmsprop":
            grad_scale = learning_rate * 0.9
        else:
            grad_scale = learning_rate * 0.5
        
        loss = np.random.random() * grad_scale * regularization
        accuracy = 1.0 - loss
        
        if epoch % 10 == 0:
            history["loss"].append(loss)
            history["val_loss"].append(loss * 1.1)
            history["accuracy"].append(accuracy)
            history["val_accuracy"].append(accuracy * 0.95)
    
    return history


def preprocess_features(
    data,
    target_col: str,
    feature_cols: list,
    normalize: bool,
    fill_strategy: str,
    clip_outliers: bool,
    outlier_threshold: float,
    encode_categoricals: bool,
):
    """Preprocess a dataset. Many params is normal for preprocessing pipelines."""
    result = data.copy()
    
    for col in feature_cols:
        if fill_strategy == "mean":
            result[col] = result[col].fillna(result[col].mean())
        elif fill_strategy == "median":
            result[col] = result[col].fillna(result[col].median())
        elif fill_strategy == "zero":
            result[col] = result[col].fillna(0)
        elif fill_strategy == "forward":
            result[col] = result[col].ffill()
        
        if clip_outliers:
            q_low = result[col].quantile(1 - outlier_threshold)
            q_high = result[col].quantile(outlier_threshold)
            result[col] = result[col].clip(q_low, q_high)
        
        if normalize:
            col_min = result[col].min()
            col_max = result[col].max()
            if col_max > col_min:
                result[col] = (result[col] - col_min) / (col_max - col_min)
    
    if encode_categoricals:
        cat_cols = result.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            result[col] = result[col].astype("category").cat.codes
    
    X = result[feature_cols].values
    y = result[target_col].values if target_col else None
    
    return X, y


def evaluate_model(
    model,
    X_test,
    y_test,
    metrics: list,
    threshold: float = 0.5,
    average: str = "macro",
) -> dict:
    """Evaluate model. Multiple metrics with different averaging is standard."""
    results = {}
    y_pred = np.random.randint(0, 2, len(y_test))
    
    for metric in metrics:
        if metric == "accuracy":
            results["accuracy"] = np.mean(y_pred == y_test)
        elif metric == "precision":
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            results["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric == "recall":
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fn = np.sum((y_pred == 0) & (y_test == 1))
            results["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "f1":
            p = results.get("precision", 0.5)
            r = results.get("recall", 0.5)
            results["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    return results
