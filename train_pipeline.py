"""
Project Sentinel - Wildlife-Vehicle Collision Predictive System
Training Pipeline: CatBoost + XGBoost + Optuna + Ridge Stacking + SHAP
"""

import os
import warnings
import logging

import numpy as np
import pandas as pd
import joblib
import shap
import optuna
from optuna.samplers import TPESampler

from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "Global Roadkill data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12,
}

CV_FOLDS = 3
OPTUNA_TRIALS = 10
RANDOM_STATE  = 42

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    log.info("Loading data from: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    log.info("Raw shape: %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────
def _parse_month(val) -> int:
    """Convert any month representation to integer 1-12."""
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    if s.isdigit():
        v = int(s)
        return v if 1 <= v <= 12 else 0
    # If it looks like a date string, parse it
    key = s[:3]
    if key in MONTH_MAP:
        return MONTH_MAP[key]
    # Try pandas parse
    try:
        return pd.to_datetime(val, errors="raise").month
    except Exception:
        return 0


def preprocess(df: pd.DataFrame):
    log.info("Preprocessing data...")

    # Fill and cast categorical columns
    if "roadType" in df.columns:
        df["roadType"] = df["roadType"].fillna("Unknown").astype(str).str.strip()
    if "class" in df.columns:
        df["class"]    = df["class"].fillna("Unknown").astype(str).str.strip()

    # Normalise month
    month_col = None
    for cname in df.columns:
        if cname.strip().lower() == "month":
            month_col = cname
            break
    if month_col is not None:
        df["month"] = df[month_col].apply(_parse_month)
    else:
        df["month"] = 0

    # Latitude / Longitude - map from dataset columns exactly
    if "decimalLatitude" in df.columns:
        df.rename(columns={"decimalLatitude": "lat"}, inplace=True)
    if "decimalLongitude" in df.columns:
        df.rename(columns={"decimalLongitude": "lon"}, inplace=True)

    # Drop rows with no spatial data
    if "lat" in df.columns and "lon" in df.columns:
        df = df.dropna(subset=["lat", "lon"])
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        
    # Handle numberOfRoadkill -> count
    if "numberOfRoadkill" in df.columns:
         df.rename(columns={"numberOfRoadkill": "count"}, inplace=True)
         df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1)
    else:
        # Fallback if somehow missing
        if "lat" in df.columns and "lon" in df.columns:
            df["count"] = df.groupby(["lat", "lon", "month"], sort=False)["month"].transform("count")
        else:
            df["count"] = 1

    df["count"] = df["count"].clip(lower=1)

    # Feature engineering
    df["season"] = df["month"].map(
        lambda m: "Winter" if m in [12,1,2] else
                  "Spring" if m in [3,4,5] else
                  "Summer" if m in [6,7,8] else
                  "Autumn" if m in [9,10,11] else "Unknown"
    )
    
    if "roadType" in df.columns:
        df["is_highway"] = df["roadType"].str.lower().str.contains(
            "highway|motorway|freeway|expressway", na=False
        ).astype(int)
    else:
        df["is_highway"] = 0

    log.info("Preprocessed shape: %s", df.shape)
    return df


def build_features(df: pd.DataFrame):
    """Return X (numeric only for XGB, with cats for CatBoost) and y."""
    feature_cols = ["month", "season", "is_highway", "roadType", "class"]
    if "lat" in df.columns:
        feature_cols += ["lat", "lon"]

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = df["count"].values.astype(float)
    return X, y


# ─────────────────────────────────────────────────────────────
# 3. CATBOOST EXPERT
# ─────────────────────────────────────────────────────────────
def train_catboost(X: pd.DataFrame, y: np.ndarray):
    log.info("Optimising CatBoostRegressor (Optuna, %d trials)...", OPTUNA_TRIALS)
    cat_features = [c for c in ["roadType", "class", "season"] if c in X.columns]

    def objective(trial):
        params = {
            "iterations":       trial.suggest_int("iterations", 200, 1000),
            "depth":            trial.suggest_int("depth", 4, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count":     trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":  trial.suggest_float("random_strength", 0.1, 10.0),
            "random_seed":      RANDOM_STATE,
            "verbose":          False,
            "allow_writing_files": False,
        }
        kf  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        mae_scores = []
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val),
                      early_stopping_rounds=30, verbose=False)
            preds = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, preds))
        return float(np.mean(mae_scores))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"random_seed": RANDOM_STATE, "verbose": False, "allow_writing_files": False})
    log.info("CatBoost best MAE: %.4f | params: %s", study.best_value, best_params)

    model = CatBoostRegressor(**best_params)
    model.fit(X, y, cat_features=cat_features, verbose=False)
    return model


# ─────────────────────────────────────────────────────────────
# 4. XGBOOST EXPERT
# ─────────────────────────────────────────────────────────────
def _encode_for_xgb(X: pd.DataFrame):
    """Label-encode categorical columns and return encoded df + encoders."""
    X_enc = X.copy()
    encoders = {}
    for col in ["roadType", "class", "season"]:
        if col in X_enc.columns:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))
            encoders[col] = le
    return X_enc.astype(float), encoders


def train_xgboost(X: pd.DataFrame, y: np.ndarray):
    log.info("Optimising XGBoostRegressor (Optuna, %d trials)...", OPTUNA_TRIALS)
    X_enc, encoders = _encode_for_xgb(X)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "tree_method":      "hist",
            "random_state":     RANDOM_STATE,
            "verbosity":        0,
            "eval_metric":      "mae",
            "early_stopping_rounds": 30,
        }
        kf  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        mae_scores = []
        for tr_idx, val_idx in kf.split(X_enc):
            X_tr, X_val = X_enc.iloc[tr_idx], X_enc.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, preds))
        return float(np.mean(mae_scores))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    best.update({
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "eval_metric": "mae",
    })
    log.info("XGBoost best MAE: %.4f | params: %s", study.best_value, best)

    model = XGBRegressor(**best)
    model.fit(X_enc, y, verbose=False)
    return model, encoders


# ─────────────────────────────────────────────────────────────
# 5. STACKING (RIDGE META-JUDGE)
# ─────────────────────────────────────────────────────────────
def build_stacking(cat_model: CatBoostRegressor,
                   xgb_model: XGBRegressor,
                   X: pd.DataFrame,
                   X_enc: pd.DataFrame,
                   y: np.ndarray,
                   cat_features: list):
    log.info("Building stacking meta-judge (Ridge)...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cat_oof = np.zeros(len(y))
    xgb_oof = np.zeros(len(y))

    for tr_idx, val_idx in kf.split(X):
        # CatBoost OOF
        cb_tmp = CatBoostRegressor(**cat_model.get_params())
        cb_tmp.set_params(verbose=False, allow_writing_files=False)
        cb_tmp.fit(X.iloc[tr_idx], y[tr_idx], cat_features=cat_features, verbose=False)
        cat_oof[val_idx] = cb_tmp.predict(X.iloc[val_idx])

        # XGBoost OOF
        xgb_tmp = XGBRegressor(**xgb_model.get_params())
        xgb_tmp.fit(X_enc.iloc[tr_idx], y[tr_idx], verbose=False)
        xgb_oof[val_idx] = xgb_tmp.predict(X_enc.iloc[val_idx])

    S = np.column_stack([cat_oof, xgb_oof])
    meta = Ridge(alpha=1.0)
    meta.fit(S, y)

    final_preds = meta.predict(S)
    mae = mean_absolute_error(y, final_preds)
    log.info("Stacked ensemble training MAE: %.4f", mae)
    return meta, mae


# ─────────────────────────────────────────────────────────────
# 6. SHAP EXPLANATIONS
# ─────────────────────────────────────────────────────────────
def compute_shap(cat_model: CatBoostRegressor, X: pd.DataFrame):
    log.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(cat_model)
    # Use a max sample for speed if the dataset is very large
    sample_size = min(5000, len(X))
    X_sample = X.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    shap_values = explainer.shap_values(X_sample)
    np.save(os.path.join(MODELS_DIR, "shap_values.npy"), shap_values)
    np.save(os.path.join(MODELS_DIR, "shap_feature_names.npy"), np.array(X_sample.columns.tolist()))
    # Also save a summary dataframe for the dashboard
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    shap_df.to_csv(os.path.join(MODELS_DIR, "shap_summary.csv"), index=False)
    log.info("SHAP values saved.")
    return shap_values, X_sample


# ─────────────────────────────────────────────────────────────
# 7. FUTURE HOTSPOT FORECASTING
# ─────────────────────────────────────────────────────────────
def forecast_hotspots(df: pd.DataFrame,
                      cat_model: CatBoostRegressor,
                      xgb_model: XGBRegressor,
                      meta: Ridge,
                      X_full: pd.DataFrame,
                      X_enc_full: pd.DataFrame,
                      y: np.ndarray):
    log.info("Forecasting future hotspots...")

    # Shift month forward by 6 months (cyclically)
    X_future = X_full.copy()
    X_future["month"] = ((X_future["month"] + 5) % 12) + 1
    X_future["season"] = X_future["month"].map(
        lambda m: "Winter" if m in [12,1,2] else
                  "Spring" if m in [3,4,5] else
                  "Summer" if m in [6,7,8] else
                  "Autumn"
    )

    X_future_enc = X_enc_full.copy()
    X_future_enc["month"] = X_future["month"]

    cat_pred_future = cat_model.predict(X_future)
    xgb_pred_future = xgb_model.predict(X_future_enc)
    S_future = np.column_stack([cat_pred_future, xgb_pred_future])
    future_scores = meta.predict(S_future)

    out = df.copy()
    out["predicted_count"]   = meta.predict(np.column_stack([cat_model.predict(X_full), xgb_model.predict(X_enc_full)]))
    out["future_risk_score"] = np.clip(future_scores, 0, None)
    out["danger_index"]      = (out["future_risk_score"] / 10.0 * 100).clip(0, 100)

    # Future hotspots: low historical count but high future risk
    hotspots = out[
        (out["count"] <= 3) &
        (out["future_risk_score"] >= 5.0)
    ].copy().sort_values("future_risk_score", ascending=False)

    hotspot_path = os.path.join(MODELS_DIR, "future_hotspots.csv")
    hotspots.to_csv(hotspot_path, index=False)
    log.info("Saved %d future hotspots to %s", len(hotspots), hotspot_path)

    processed_path = os.path.join(MODELS_DIR, "preprocessed_data.csv")
    out.to_csv(processed_path, index=False)
    log.info("Saved preprocessed data to %s", processed_path)
    return hotspots, out


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    df = preprocess(df)

    # Feature matrix
    X, y = build_features(df)
    cat_features = [c for c in ["roadType", "class", "season"] if c in X.columns]
    X_enc, label_encoders = _encode_for_xgb(X)

    # Train Model A: CatBoost
    cat_model = train_catboost(X, y)
    joblib.dump(cat_model, os.path.join(MODELS_DIR, "cat_expert.pkl"))
    log.info("Saved cat_expert.pkl")

    # Train Model B: XGBoost
    xgb_model, label_encoders = train_xgboost(X, y)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_expert.pkl"))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    log.info("Saved xgb_expert.pkl")

    # Stacking Meta-Judge
    meta_judge, final_mae = build_stacking(
        cat_model, xgb_model, X, X_enc, y, cat_features
    )
    joblib.dump(meta_judge, os.path.join(MODELS_DIR, "meta_judge.pkl"))
    log.info("Saved meta_judge.pkl | Final Stacked MAE: %.4f", final_mae)

    # SHAP
    compute_shap(cat_model, X)

    # Future Forecasting
    forecast_hotspots(df, cat_model, xgb_model, meta_judge, X, X_enc, y)

    log.info("Training complete. All models and artifacts saved to: %s", MODELS_DIR)
    print(f"\n[DONE] Final Stacked MAE: {final_mae:.4f}")
    print(f"[DONE] Models directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()
