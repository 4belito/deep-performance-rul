"""Operating-condition normalization + residual aggregation + health index pipeline"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


# ============================================================
# Residual aggregation transformer
# ============================================================
class OperCondResidualAggregator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        oc_pipe: Pipeline,
        opcond_cols: list[str],
        perform_cols: list[str],
        unit_col: str = "unit",
        cycle_col: str = "cycle",
    ):
        self.oc_pipe = oc_pipe
        self.opcond_cols = opcond_cols
        self.perform_cols = perform_cols
        self.unit_col = unit_col
        self.cycle_col = cycle_col

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self.oc_pipe)

        X = df[self.opcond_cols]
        perf_nom = self.oc_pipe.predict(X)

        res = pd.DataFrame(
            df[self.perform_cols].values - perf_nom,
            columns=self.perform_cols,
            index=df.index,
        )
        res[self.unit_col] = df[self.unit_col].values
        res[self.cycle_col] = df[self.cycle_col].values

        return res.groupby([self.unit_col, self.cycle_col], as_index=False).mean()


# ============================================================
# Health Index transformer
# ============================================================
class HealthIndexTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        metrics: list[str],
        cycle_col: str = "cycle",
        q_low: float = 0.01,
        q_high: float = 0.99,
        corr_thresh: float = 0.6,
        range_thresh: float = 0.6,
    ):
        self.metrics = metrics
        self.cycle_col = cycle_col
        self.q_low = q_low
        self.q_high = q_high
        self.corr_thresh = corr_thresh
        self.range_thresh = range_thresh

    # --------------------------------------------------
    # Metrics kept after pruning
    # --------------------------------------------------
    def get_performances(self):
        check_is_fitted(self, "signs_")
        return [m for m, s in self.signs_.items() if s != 0]

    # --------------------------------------------------
    # Fit: monotonicity + bounds
    # --------------------------------------------------
    def fit(self, df: pd.DataFrame, y=None):
        self.signs_ = {}
        self.bounds_ = {}
        self.range_scores_ = {}

        # 1️⃣ Monotonicity screening
        for m in self.metrics:
            corr = df[m].corr(df[self.cycle_col], method="spearman")

            if abs(corr) < self.corr_thresh:
                self.signs_[m] = 0
            else:
                self.signs_[m] = np.sign(corr)

        # 2️⃣ Compute normalization bounds
        self.set_bounds(df, self.q_low, self.q_high)

        return self

    # --------------------------------------------------
    # Learn quantile bounds
    # --------------------------------------------------
    def set_bounds(self, df: pd.DataFrame, q_low: float, q_high: float):
        check_is_fitted(self, "signs_")

        self.bounds_ = {}

        for m in self.metrics:
            sign = self.signs_.get(m, 0)
            if sign == 0:
                continue

            r = sign * df[m].to_numpy()
            lo = np.quantile(r, q_low)
            hi = np.quantile(r, q_high)

            if hi > lo:
                self.bounds_[m] = (lo, hi)
            else:
                self.signs_[m] = 0

        return self

    # --------------------------------------------------
    # Transform: normalize + prune by range
    # --------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["signs_", "bounds_"])

        out = df.copy()

        # 1️⃣ Normalize
        for m in self.metrics:
            sign = self.signs_.get(m, 0)

            if sign == 0:
                continue

            lo, hi = self.bounds_[m]
            r = sign * out[m].to_numpy()
            out[m] = 1.0 - np.clip((r - lo) / (hi - lo), 0.0, 1.0)
        return out

    def prune_performances_by_range(self, df_norm: pd.DataFrame):
        active_metrics = list(self.get_performances())

        for m in active_metrics:
            unit_ranges = []

            for _, g in df_norm.groupby("unit"):
                unit_ranges.append(g[m].max() - g[m].min())

            avg_range = np.mean(unit_ranges)
            self.range_scores_[m] = avg_range  # useful for debugging

            if avg_range < self.range_thresh:
                self.signs_[m] = 0


# ============================================================
# Full Estimation Pipeline
# ============================================================
class EstimationPipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end PHM pipeline with a single fit():
    - fits OC model on healthy data
    - computes residuals
    - learns health indices
    """

    def __init__(
        self,
        oc_pipe: Pipeline,
        residual_aggregator: OperCondResidualAggregator,
        hi_transformer: HealthIndexTransformer,
        hs_col: str = "hs",
    ):
        self.oc_pipe = oc_pipe
        self.residual_aggregator = residual_aggregator
        self.hi_transformer = hi_transformer
        self.hs_col = hs_col

    def get_performances(self):
        check_is_fitted(self, "is_fitted_")
        return self.hi_transformer.get_performances()

    def prune_performances_by_range(self, df_norm: pd.DataFrame):
        check_is_fitted(self, "is_fitted_")
        return self.hi_transformer.prune_performances_by_range(df_norm)

    def fit(self, df: pd.DataFrame, y=None):
        # --- Stage 1: OC normalization (healthy only)
        df_h = df[df[self.hs_col] == 1.0]

        self.oc_pipe.fit(
            df_h[self.residual_aggregator.opcond_cols],
            df_h[self.residual_aggregator.perform_cols].values.astype(np.float32),
        )

        # inject fitted OC model
        self.residual_aggregator.oc_pipe = self.oc_pipe

        # --- Stage 2: residuals on full data
        residuals = self.residual_aggregator.transform(df)

        # --- Stage 3: health index learning
        self.hi_transformer.fit(residuals)

        self.is_fitted_ = True
        return self

    def set_bounds(self, df: pd.DataFrame, q_low: float = 0, q_high: float = 1.0):
        check_is_fitted(self, "is_fitted_")

        residuals = self.residual_aggregator.transform(df)
        self.hi_transformer.set_bounds(residuals, q_low, q_high)
        return self

    def transform(self, df: pd.DataFrame):
        check_is_fitted(self, "is_fitted_")

        residuals = self.residual_aggregator.transform(df)
        return self.hi_transformer.transform(residuals)
