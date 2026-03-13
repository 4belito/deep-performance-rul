"""Operating-condition normalization + residual aggregation + health index pipeline"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


# ============================================================
# Residual aggregation transformer
# ============================================================
class OperCondResidualAggregator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        opcond_cols: list[str],
        perform_cols: list[str],
        unit_col: str = "unit",
        cycle_col: str = "cycle",
    ):
        self.opcond_cols = opcond_cols
        self.perform_cols = perform_cols
        self.unit_col = unit_col
        self.cycle_col = cycle_col

    def fit(self, X, y=None):
        return self

    def transform(
        self, df: pd.DataFrame, oc_pipe: Pipeline, y_scaler: StandardScaler
    ) -> pd.DataFrame:
        check_is_fitted(oc_pipe)

        X = df[self.opcond_cols]
        perf_nom_scaled = oc_pipe.predict(X)
        perf_nom = y_scaler.inverse_transform(perf_nom_scaled)

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
        performs: list[str],
        cycle_col: str = "cycle",
        nom_margin: float = 0.01,
        eol_margin: float = 0.01,
    ):
        self.performs = performs
        self.cycle_col = cycle_col
        self.eol_margin = eol_margin
        self.nom_margin = nom_margin

    # --------------------------------------------------
    # Fit: monotonicity + bounds
    # --------------------------------------------------
    def fit(self, df: pd.DataFrame, y=None):
        self.bounds_ = {}
        self.signs_ = {}
        self.range_scores_ = {}
        self.corr_scores_ = {}

        # 1️⃣ Monotonicity screening
        df_norm = df.copy()
        for m in self.performs:
            corr = df[m].corr(df[self.cycle_col], method="spearman")
            sign = np.sign(corr)
            r = np.sign(corr) * df[m].to_numpy()
            lo = np.quantile(r, self.nom_margin)
            hi = np.quantile(r, 1 - self.eol_margin)
            if hi <= lo:
                continue
            s = 1.0 - np.clip((r - lo) / (hi - lo), 0.0, 1.0)

            self.bounds_[m] = (lo, hi)
            self.signs_[m] = sign
            df_norm[m] = s

        # performance selection by range
        groups = list(df_norm.groupby("unit"))
        for m in self.performs:
            unit_ranges = [g[m].max() - g[m].min() for _, g in groups]
            avg_range = np.mean(unit_ranges)
            self.range_scores_[m] = avg_range
            self.corr_scores_[m] = df_norm[m].corr(df[self.cycle_col], method="spearman")
        return self

    def get_valid_performances(self, min_range: float = 0.0, min_corr: float = 0.0):
        check_is_fitted(self, "range_scores_")
        performances = []
        for m in self.performs:
            if self.range_scores_[m] > min_range and abs(self.corr_scores_[m]) > min_corr:
                performances.append(m)
        return performances

    # --------------------------------------------------
    # Transform: normalize + prune by range
    # --------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["bounds_", "signs_"])

        out = df.copy()

        # 1️⃣ Normalize
        for m in self.performs:
            lo, hi = self.bounds_[m]
            sign = self.signs_[m]
            r = sign * out[m].to_numpy()
            out[m] = 1.0 - np.clip((r - lo) / (hi - lo), 0.0, 1.0)
        return out


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
        self.y_scaler = StandardScaler()

    def get_performances(self):
        check_is_fitted(self, "is_fitted_")
        return self.residual_aggregator.perform_cols

    def get_valid_performances(self, min_range: float = 0.0, min_corr: float = 0.0):
        check_is_fitted(self, "is_fitted_")
        return self.hi_transformer.get_valid_performances(min_range, min_corr)

    def fit(self, df: pd.DataFrame, y=None):
        # --- Stage 1: OC normalization (healthy only)
        df_h = df[df[self.hs_col] == 1.0]
        Y = df_h[self.residual_aggregator.perform_cols].values
        Y_scaled = self.y_scaler.fit_transform(Y)
        self.oc_pipe.fit(
            df_h[self.residual_aggregator.opcond_cols],
            Y_scaled.astype(np.float32),
        )
        # --- Stage 2: residuals on full data
        residuals = self.residual_aggregator.transform(df, self.oc_pipe, self.y_scaler)

        # --- Stage 3: health index learning
        self.hi_transformer.fit(residuals)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame):
        check_is_fitted(self, "is_fitted_")

        residuals = self.residual_aggregator.transform(df, self.oc_pipe, self.y_scaler)
        return self.hi_transformer.transform(residuals)
