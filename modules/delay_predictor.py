"""
delay_predictor.py  — Random Forest model that predicts per-station train delays.

Features used:
  train_id_enc, station_enc, zone_enc,
  scheduled_hour, day_of_week, month, historical_avg_delay

Output: (predicted_delay_min: float, confidence: float 0-1)
"""

from __future__ import annotations

import os
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


class DelayPredictor:
    MODEL_FILE = "delay_model.pkl"
    META_FILE  = "delay_meta.pkl"

    def __init__(self, model_dir: str, data_dir: str):
        self.model_dir = model_dir
        self.data_dir  = data_dir
        self.model: Optional[RandomForestRegressor] = None
        self.train_enc = LabelEncoder()
        self.stn_enc   = LabelEncoder()
        self.zone_enc  = LabelEncoder()
        self._station_avg: Dict[str, float] = {}
        self._trained = False

    # ── load or train ─────────────────────────────────────────────────────────

    def load_or_train(self, force: bool = False) -> bool:
        """Return True if model was loaded from disk, False if freshly trained."""
        model_path = os.path.join(self.model_dir, self.MODEL_FILE)
        meta_path  = os.path.join(self.model_dir, self.META_FILE)

        if not force and os.path.exists(model_path) and os.path.exists(meta_path):
            self.model = joblib.load(model_path)
            meta = joblib.load(meta_path)
            self.train_enc      = meta["train_enc"]
            self.stn_enc        = meta["stn_enc"]
            self.zone_enc       = meta["zone_enc"]
            self._station_avg   = meta["station_avg"]
            self._trained = True
            return True  # loaded from cache

        self._train_model()
        return False  # freshly trained

    def _train_model(self):
        path = os.path.join(self.data_dir, "historical_delays.csv")
        if not os.path.exists(path):
            raise FileNotFoundError("historical_delays.csv not found. Run generate-data first.")

        df = pd.read_csv(path)
        self._station_avg = df.groupby("station_code")["actual_delay_min"].mean().to_dict()

        # Encode categoricals
        self.train_enc.fit(df["train_id"].astype(str))
        self.stn_enc.fit(df["station_code"].astype(str))
        self.zone_enc.fit(df["zone"].astype(str))

        X = self._make_features(df)
        y = df["actual_delay_min"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Compute metrics
        y_pred = self.model.predict(X_test)
        self._mae  = mean_absolute_error(y_test, y_pred)
        self._rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        self._r2   = self.model.score(X_test, y_test)

        # Persist
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(self.model_dir, self.MODEL_FILE))
        joblib.dump(
            {
                "train_enc":   self.train_enc,
                "stn_enc":     self.stn_enc,
                "zone_enc":    self.zone_enc,
                "station_avg": self._station_avg,
            },
            os.path.join(self.model_dir, self.META_FILE),
        )
        self._trained = True

    def retrain(self) -> Dict[str, float]:
        """Force retrain and return metrics."""
        self._train_model()
        return {"mae": self._mae, "rmse": self._rmse, "r2": self._r2}

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        train_id: int,
        station_code: str,
        scheduled_hour: int,
        day_of_week: int,
        month: int,
        zone: str = "NR",
    ) -> Tuple[float, float]:
        """
        Returns (predicted_delay_minutes, confidence_score 0-1).
        Confidence is derived from the variance across trees.
        """
        if not self._trained or self.model is None:
            self.load_or_train()

        # Handle unseen labels gracefully
        tid_str = str(train_id)
        stn_str = str(station_code)
        zon_str = str(zone)

        if tid_str not in self.train_enc.classes_:
            tid_enc = 0
        else:
            tid_enc = int(self.train_enc.transform([tid_str])[0])

        if stn_str not in self.stn_enc.classes_:
            stn_enc = 0
        else:
            stn_enc = int(self.stn_enc.transform([stn_str])[0])

        if zon_str not in self.zone_enc.classes_:
            zon_enc = 0
        else:
            zon_enc = int(self.zone_enc.transform([zon_str])[0])

        avg_delay = self._station_avg.get(station_code, 12.0)

        row = np.array([[
            tid_enc, stn_enc, zon_enc,
            scheduled_hour, day_of_week, month, avg_delay
        ]])

        # Gather per-tree predictions for confidence
        tree_preds = np.array([t.predict(row)[0] for t in self.model.estimators_])
        pred_mean  = float(tree_preds.mean())
        pred_std   = float(tree_preds.std())

        # Confidence: low variance → high confidence
        max_std    = 20.0
        confidence = float(max(0.0, min(1.0, 1.0 - (pred_std / max_std))))

        return round(max(0.0, pred_mean), 1), round(confidence, 2)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_features(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            self.train_enc.transform(df["train_id"].astype(str)),
            self.stn_enc.transform(df["station_code"].astype(str)),
            self.zone_enc.transform(df["zone"].astype(str)),
            df["scheduled_hour"].values,
            df["day_of_week"].values,
            df["month"].values,
            df["historical_avg_delay"].values,
        ])

    @property
    def metrics(self) -> Dict[str, float]:
        return {
            "mae":  getattr(self, "_mae",  None),
            "rmse": getattr(self, "_rmse", None),
            "r2":   getattr(self, "_r2",   None),
        }
