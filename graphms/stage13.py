"""Inference wrapper for the frozen refreshed Stage13 SVM/Ridge artifacts."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

EXPECTED_RUN_FINGERPRINT = "0883979e8ef5f313f231ed08fa8e0a7d419c4cddb16cf79b9d87b9076da270be"
EXPECTED_STAGE13_VERSION = "GraphMSNet-Stage13-HYBRID-REFRESH-v1"

def _load(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise RuntimeError(f"Unexpected Stage13 artifact type: {type(bundle)}")
    if bundle.get("run_fingerprint") != EXPECTED_RUN_FINGERPRINT:
        raise RuntimeError("Stage13 run fingerprint mismatch")
    if bundle.get("stage13_version") != EXPECTED_STAGE13_VERSION:
        raise RuntimeError("Stage13 version mismatch")
    return bundle

def predict_stage13(
    feature_row: pd.DataFrame,
    classifier_path: str | Path,
    regressor_path: str | Path,
) -> dict:
    if len(feature_row) != 1:
        raise ValueError("predict_stage13 expects exactly one Stage12 feature row")
    cb = _load(classifier_path)
    rb = _load(regressor_path)
    if cb.get("primary_family") != "MRI_SPATIAL_SVM" or cb.get("feature_set") != "mri_spatial":
        raise RuntimeError("Classifier identity mismatch")
    if rb.get("primary_family") != "MRI_SPATIAL_RIDGE" or rb.get("feature_set") != "mri_spatial":
        raise RuntimeError("Regressor identity mismatch")

    ccols = list(cb["feature_columns"])
    rcols = list(rb["feature_columns"])
    missing = sorted((set(ccols) | set(rcols)) - set(feature_row.columns))
    if missing:
        raise RuntimeError(f"Stage12 feature schema missing columns: {missing[:20]}")

    class_model = cb["model"]
    if class_model.get("kind") != "linear_svm":
        raise RuntimeError("Frozen classifier inner model is not linear_svm")
    raw_score = float(np.asarray(
        class_model["pipeline"].decision_function(feature_row[ccols]), dtype=float
    ).reshape(-1)[0])
    probability = float(
        cb["calibrator"].predict_proba(np.asarray([[raw_score]], dtype=float))[0, 1]
    )

    reg_model = rb["model"]
    if reg_model.get("kind") != "ridge":
        raise RuntimeError("Frozen regressor inner model is not ridge")
    edss = float(np.asarray(
        reg_model["pipeline"].predict(feature_row[rcols]), dtype=float
    ).reshape(-1)[0])
    edss = float(np.clip(edss, 0.0, 10.0))

    thresholds = cb["thresholds"]
    return {
        "edss_ge4_probability": probability,
        "edss_ge4_balanced_threshold": float(thresholds["balanced_threshold"]),
        "edss_ge4_balanced_prediction": int(probability >= float(thresholds["balanced_threshold"])),
        "edss_ge4_high_sensitivity_threshold": float(thresholds["high_sensitivity_threshold"]),
        "edss_ge4_high_sensitivity_prediction": int(
            probability >= float(thresholds["high_sensitivity_threshold"])
        ),
        "predicted_edss": edss,
        "classifier": "MRI_SPATIAL_SVM",
        "regressor": "MRI_SPATIAL_RIDGE",
        "feature_set": "mri_spatial",
        "claim_scope": "development patient-grouped CV",
    }
