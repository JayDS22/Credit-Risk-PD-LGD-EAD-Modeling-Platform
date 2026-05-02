"""
Loss Given Default (LGD) Model
Beta regression and two-stage approach for recovery rate estimation.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class BetaRegression:
    """Beta regression for bounded (0,1) LGD estimation."""

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.phi = None  # precision parameter

    @staticmethod
    def _logit(p):
        return np.log(p / (1 - p))

    @staticmethod
    def _inv_logit(x):
        return 1 / (1 + np.exp(-x))

    def _neg_log_likelihood(self, params, X, y):
        """Negative log-likelihood for beta regression."""
        n_features = X.shape[1]
        beta = params[:n_features + 1]
        phi = np.exp(params[-1])  # ensure positive

        # Linear predictor
        eta = beta[0] + X @ beta[1:]
        mu = self._inv_logit(eta)

        # Clamp mu to avoid boundary issues
        mu = np.clip(mu, 1e-6, 1 - 1e-6)

        # Beta distribution parameters
        a = mu * phi
        b = (1 - mu) * phi

        # Log-likelihood
        ll = np.sum(
            stats.beta.logpdf(y, a, b)
        )
        return -ll

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit beta regression model."""
        # Clamp y to (0,1) open interval
        y_clamped = np.clip(y, 1e-4, 1 - 1e-4)

        n_features = X.shape[1]
        # Initial params: [intercept, betas..., log_phi]
        x0 = np.zeros(n_features + 2)
        x0[0] = self._logit(np.mean(y_clamped))
        x0[-1] = np.log(5.0)

        try:
            result = optimize.minimize(
                self._neg_log_likelihood,
                x0,
                args=(X, y_clamped),
                method='L-BFGS-B',
                options={'maxiter': 500}
            )
            self.intercept = result.x[0]
            self.coefficients = result.x[1:n_features + 1]
            self.phi = np.exp(result.x[-1])
        except Exception as e:
            logger.warning(f"Beta regression optimization failed: {e}. Using OLS fallback.")
            lr = LinearRegression()
            lr.fit(X, self._logit(y_clamped))
            self.intercept = lr.intercept_
            self.coefficients = lr.coef_
            self.phi = 5.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict LGD values."""
        eta = self.intercept + X @ self.coefficients
        return np.clip(self._inv_logit(eta), 0.01, 0.99)


class LGDModel:
    """Two-stage LGD model: cure/no-cure classification + severity regression."""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.beta_model = BetaRegression()
        self.severity_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42
        )
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        self._is_fitted = False

    def _get_lgd_features(self) -> List[str]:
        """LGD-specific features."""
        return [
            'credit_score', 'ltv_ratio', 'property_value', 'loan_amount',
            'credit_utilization', 'dti_ratio', 'unemployment_rate',
            'hpi_change', 'interest_rate', 'loan_term_months',
            'employment_years', 'num_delinquencies_2y',
            'credit_history_months', 'annual_income',
            'purpose_encoded', 'home_ownership_encoded', 'region_encoded'
        ]

    def fit(self, df: pd.DataFrame) -> Dict:
        """Train LGD model on defaulted accounts."""
        # Filter to defaulted accounts
        default_df = df[df['default_flag'] == 1].copy()
        if len(default_df) == 0:
            logger.error("No defaulted accounts for LGD modeling")
            return {}

        logger.info(f"Training LGD model on {len(default_df):,} defaulted accounts")

        # Get features
        available_features = [f for f in self._get_lgd_features() if f in default_df.columns]
        self.feature_names = available_features

        X = default_df[self.feature_names].copy()
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], 0)
        y = default_df['lgd'].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Stage 1: Beta Regression
        logger.info("Fitting beta regression for LGD...")
        self.beta_model.fit(X_train_scaled, y_train)
        beta_preds = self.beta_model.predict(X_test_scaled)

        # Stage 2: Gradient Boosting for residuals
        logger.info("Fitting severity model...")
        self.severity_model.fit(X_train, y_train)
        gb_preds = self.severity_model.predict(X_test)
        gb_preds = np.clip(gb_preds, 0.01, 0.99)

        # Ensemble: weighted average
        ensemble_preds = 0.4 * beta_preds + 0.6 * gb_preds
        ensemble_preds = np.clip(ensemble_preds, 0.01, 0.99)

        # Metrics
        self.metrics = {
            'r2': round(r2_score(y_test, ensemble_preds), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_test, ensemble_preds)), 4),
            'mae': round(mean_absolute_error(y_test, ensemble_preds), 4),
            'mean_lgd_actual': round(np.mean(y_test), 4),
            'mean_lgd_predicted': round(np.mean(ensemble_preds), 4),
            'beta_r2': round(r2_score(y_test, beta_preds), 4),
            'gb_r2': round(r2_score(y_test, gb_preds), 4),
            'n_samples': len(default_df)
        }

        self._is_fitted = True
        self._test_actual = y_test
        self._test_predicted = ensemble_preds

        logger.info(f"LGD Model - R2: {self.metrics['r2']:.4f}, "
                     f"RMSE: {self.metrics['rmse']:.4f}")
        return self.metrics

    def predict_lgd(self, X: pd.DataFrame) -> np.ndarray:
        """Predict LGD for given features."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_feat = X[self.feature_names].copy()
        X_feat = X_feat.fillna(X_feat.median())
        X_feat = X_feat.replace([np.inf, -np.inf], 0)

        X_scaled = self.scaler.transform(X_feat)
        beta_preds = self.beta_model.predict(X_scaled)
        gb_preds = np.clip(self.severity_model.predict(X_feat), 0.01, 0.99)

        return np.clip(0.4 * beta_preds + 0.6 * gb_preds, 0.01, 0.99)

    def compute_downturn_lgd(self, lgd_pit: np.ndarray,
                              stress_multiplier: float = 1.3) -> np.ndarray:
        """Compute downturn LGD per Basel III requirements."""
        lgd_dt = np.clip(lgd_pit * stress_multiplier, 0.10, 1.0)
        floor = self.config.get('lgd_downturn_floor', 0.10)
        return np.maximum(lgd_dt, floor)
