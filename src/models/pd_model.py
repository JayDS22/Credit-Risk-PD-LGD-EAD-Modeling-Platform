"""
Probability of Default (PD) Model
Logistic Regression and XGBoost with TTC/PIT calibration
using Vasicek single-factor model.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    precision_recall_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, Tuple, Optional, List
import logging
import warnings
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PDModel:
    """Probability of Default estimation with Basel III calibration."""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.logistic_model = None
        self.xgb_model = None
        self.active_model = None
        self.calibrator = None
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        self.risk_grade_boundaries: List[float] = []

    def _build_logistic(self) -> LogisticRegression:
        """Build logistic regression model."""
        params = self.config.get('logistic_params', {})
        return LogisticRegression(
            C=params.get('C', 1.0),
            max_iter=params.get('max_iter', 1000),
            solver=params.get('solver', 'lbfgs'),
            class_weight='balanced',
            random_state=42
        )

    def _build_xgboost(self) -> xgb.XGBClassifier:
        """Build XGBoost classifier."""
        params = self.config.get('xgboost_params', {})
        return xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            min_child_weight=params.get('min_child_weight', 5),
            reg_alpha=params.get('reg_alpha', 0.1),
            reg_lambda=params.get('reg_lambda', 1.0),
            scale_pos_weight=params.get('scale_pos_weight', 20),
            eval_metric='auc',
            random_state=42,
            use_label_encoder=False
        )

    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_names: List[str]) -> Dict:
        """Train PD models and evaluate performance."""
        self.feature_names = feature_names
        X_feat = X[feature_names].copy()

        # Handle missing values
        X_feat = X_feat.fillna(X_feat.median())
        X_feat = X_feat.replace([np.inf, -np.inf], 0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Logistic Regression
        logger.info("Training Logistic Regression PD model...")
        self.logistic_model = self._build_logistic()
        self.logistic_model.fit(X_train_scaled, y_train)
        lr_probs = self.logistic_model.predict_proba(X_test_scaled)[:, 1]
        lr_metrics = self._compute_metrics(y_test, lr_probs, "Logistic")

        # Train XGBoost
        logger.info("Training XGBoost PD model...")
        self.xgb_model = self._build_xgboost()
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        xgb_probs = self.xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = self._compute_metrics(y_test, xgb_probs, "XGBoost")

        # Select best model
        if xgb_metrics['auc'] >= lr_metrics['auc']:
            self.active_model = 'xgboost'
            self.metrics = xgb_metrics
            best_probs = xgb_probs
            logger.info(f"Selected XGBoost (AUC: {xgb_metrics['auc']:.4f})")
        else:
            self.active_model = 'logistic'
            self.metrics = lr_metrics
            best_probs = lr_probs
            logger.info(f"Selected Logistic (AUC: {lr_metrics['auc']:.4f})")

        # Build risk grades
        self._build_risk_grades(best_probs, y_test)

        # Store test data for later use
        self._test_probs = best_probs
        self._test_actual = y_test

        return {
            'logistic': lr_metrics,
            'xgboost': xgb_metrics,
            'selected': self.active_model,
            'metrics': self.metrics
        }

    def _compute_metrics(self, y_true: pd.Series, y_prob: np.ndarray,
                          model_name: str) -> Dict:
        """Compute comprehensive PD model metrics."""
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1

        # KS Statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        ks_stat = np.max(tpr - fpr)

        # Brier Score
        brier = brier_score_loss(y_true, y_prob)

        # Log Loss
        from sklearn.metrics import log_loss
        ll = log_loss(y_true, y_prob)

        metrics = {
            'auc': round(auc, 4),
            'gini': round(gini, 4),
            'ks_statistic': round(ks_stat, 4),
            'brier_score': round(brier, 6),
            'log_loss': round(ll, 6),
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }

        logger.info(f"{model_name} - AUC: {auc:.4f}, Gini: {gini:.4f}, KS: {ks_stat:.4f}")
        return metrics

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of default."""
        X_feat = X[self.feature_names].copy()
        X_feat = X_feat.fillna(X_feat.median())
        X_feat = X_feat.replace([np.inf, -np.inf], 0)

        if self.active_model == 'xgboost':
            return self.xgb_model.predict_proba(X_feat)[:, 1]
        else:
            X_scaled = self.scaler.transform(X_feat)
            return self.logistic_model.predict_proba(X_scaled)[:, 1]

    def _build_risk_grades(self, probs: np.ndarray, y_true: pd.Series,
                            n_grades: int = 10):
        """Build risk grade system using predicted probabilities."""
        self.risk_grade_boundaries = np.percentile(
            probs, np.linspace(0, 100, n_grades + 1)
        ).tolist()

        grades = pd.cut(probs, bins=self.risk_grade_boundaries,
                         labels=[f'Grade_{i+1}' for i in range(n_grades)],
                         include_lowest=True)

        grade_stats = pd.DataFrame({
            'grade': grades, 'pd': probs, 'default': y_true
        }).groupby('grade', observed=True).agg(
            count=('pd', 'size'),
            mean_pd=('pd', 'mean'),
            actual_dr=('default', 'mean')
        ).reset_index()

        self.risk_grade_stats = grade_stats
        if len(grade_stats) >= 2:
            separation = (
                grade_stats['actual_dr'].max() / max(grade_stats['actual_dr'].min(), 1e-6)
            )
            logger.info(f"Risk grade separation (worst/best): {separation:.1f}x")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the active model."""
        if self.active_model == 'xgboost' and self.xgb_model:
            importance = self.xgb_model.feature_importances_
        elif self.logistic_model:
            importance = np.abs(self.logistic_model.coef_[0])
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        df['importance_pct'] = (df['importance'] / df['importance'].sum() * 100).round(2)
        return df


class VasicekCalibrator:
    """Through-the-Cycle (TTC) and Point-in-Time (PIT) PD calibration
    using the Vasicek single-factor model."""

    def __init__(self, asset_correlation_range: Tuple[float, float] = (0.15, 0.24)):
        self.rho_range = asset_correlation_range
        self.calibrated_pds: Dict[str, np.ndarray] = {}

    def estimate_asset_correlation(self, pd_ttc: float) -> float:
        """Estimate asset correlation per Basel II/III formula."""
        # Basel correlation formula for corporate exposures
        rho = (
            0.12 * (1 - np.exp(-50 * pd_ttc)) / (1 - np.exp(-50)) +
            0.24 * (1 - (1 - np.exp(-50 * pd_ttc)) / (1 - np.exp(-50)))
        )
        return np.clip(rho, self.rho_range[0], self.rho_range[1])

    def ttc_to_pit(self, pd_ttc: np.ndarray,
                    macro_factor: float = 0.0) -> np.ndarray:
        """Convert TTC PD to PIT PD using Vasicek model.

        Args:
            pd_ttc: Through-the-cycle PD estimates
            macro_factor: Systematic factor Z ~ N(0,1), negative = downturn
        """
        pit_pds = np.zeros_like(pd_ttc)
        for i, pd_val in enumerate(pd_ttc):
            rho = self.estimate_asset_correlation(pd_val)
            # Vasicek conditional default probability
            z_threshold = stats.norm.ppf(pd_val)
            pit_pd = stats.norm.cdf(
                (z_threshold - np.sqrt(rho) * macro_factor) / np.sqrt(1 - rho)
            )
            pit_pds[i] = pit_pd

        self.calibrated_pds['pit'] = pit_pds
        return pit_pds

    def pit_to_ttc(self, pd_pit: np.ndarray,
                    macro_factor: float = 0.0) -> np.ndarray:
        """Convert PIT PD back to TTC PD."""
        ttc_pds = np.zeros_like(pd_pit)
        for i, pd_val in enumerate(pd_pit):
            rho = self.estimate_asset_correlation(pd_val)
            z_conditional = stats.norm.ppf(pd_val)
            ttc_pd = stats.norm.cdf(
                z_conditional * np.sqrt(1 - rho) + np.sqrt(rho) * macro_factor
            )
            ttc_pds[i] = ttc_pd

        self.calibrated_pds['ttc'] = ttc_pds
        return ttc_pds

    def stress_pd(self, pd_base: np.ndarray,
                   stress_factor: float = -2.0) -> np.ndarray:
        """Apply stress scenario to PD estimates."""
        return self.ttc_to_pit(pd_base, macro_factor=stress_factor)
