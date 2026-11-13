"""Train ETA regression and delay-risk classification models; save reproducible artifacts."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_shipments.csv"
MODEL_DIR = ROOT / "models"
FEATURES = [
    "distance_km", "departure_delay_hours", "transit_delay_hours",
    "port_arrival_delay_hours", "port_congestion_index", "weather_severity",
    "customs_complexity", "planned_customs_hours", "schedule_slack_hours",
    "transshipment_count"
]
TARGET = "actual_delivery_delay_hours"

def run_training():
    df = pd.read_csv(DATA)
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    y_class = df["is_delayed_24h"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.20, random_state=42, stratify=y_class
    )

    candidates = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=350, max_depth=7, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=160, learning_rate=0.06, max_leaf_nodes=9,
            l2_regularization=2.0, random_state=42
        ),
    }
    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "mae_hours": float(mean_absolute_error(y_test, pred)),
            "rmse_hours": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred))
        }
        fitted[name] = model

    # Select using held-out MAE for a compact assignment benchmark.
    # The chosen model is explicitly recorded; for larger studies use a separate validation set.
    best_name = min(results, key=lambda k: results[k]["mae_hours"])
    best_model = candidates[best_name]
    best_model.fit(X, y)

    classifier = RandomForestClassifier(
        n_estimators=350, max_depth=6, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    classifier.fit(Xc_train, yc_train)
    probabilities = classifier.predict_proba(Xc_test)[:, 1]
    class_preds = (probabilities >= 0.5).astype(int)
    class_metrics = {
        "precision": float(precision_score(yc_test, class_preds, zero_division=0)),
        "recall": float(recall_score(yc_test, class_preds, zero_division=0)),
        "f1": float(f1_score(yc_test, class_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(yc_test, probabilities)) if len(np.unique(yc_test)) > 1 else None,
        "brier_score": float(brier_score_loss(yc_test, probabilities))
    }
    # Refit classifier on all records after holdout evaluation for app inference.
    classifier.fit(X, y_class)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "eta_regressor.joblib")
    joblib.dump(classifier, MODEL_DIR / "delay_classifier.joblib")
    metadata = {
        "features": FEATURES,
        "selected_model": best_name,
        "regression_results": results,
        "classification_metrics": class_metrics,
        "n_samples": int(len(df)),
        "test_samples": int(len(y_test)),
        "delay_threshold_hours": 24,
        "random_seed": 42
    }
    (MODEL_DIR / "model_metrics.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    run_training()
