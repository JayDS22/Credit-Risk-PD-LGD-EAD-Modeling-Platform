"""
Exposure at Default (EAD) Model
Cox regression survival analysis with credit conversion factors.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import Dict, List, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CoxCCFModel:
    """Credit Conversion Factor model using Cox-inspired approach."""

    def __init__(self):
        self.coefficients: Optional[np.ndarray] = None
        self.baseline_hazard: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self._is_fitted = False

    def _partial_likelihood(self, beta, X, T, E):
        """Compute negative partial log-likelihood."""
        risk_scores = X @ beta
        ll = 0.0

        events = np.where(E == 1)[0]
        for i in events:
            at_risk = T >= T[i]
            ll += risk_scores[i] - np.log(np.sum(np.exp(risk_scores[at_risk])) + 1e-10)

        return -ll

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray):
        """Fit Cox-type model for EAD estimation.

        Args:
            X: Feature matrix
            T: Time to event (months to default or censoring)
            E: Event indicator (1=default, 0=censored)
        """
        from scipy.optimize import minimize

        n_features = X.shape[1]
        x0 = np.zeros(n_features)

        try:
            result = minimize(
                self._partial_likelihood,
                x0,
                args=(X, T, E),
                method='L-BFGS-B',
                options={'maxiter': 200}
            )
            self.coefficients = result.x
        except Exception as e:
            logger.warning(f"Cox optimization failed: {e}. Using linear fallback.")
            lr = LinearRegression()
            lr.fit(X[E == 1], T[E == 1])
            self.coefficients = lr.coef_

        self._compute_baseline_hazard(X, T, E)
        self._is_fitted = True

    def _compute_baseline_hazard(self, X, T, E):
        """Compute Breslow baseline hazard estimate."""
        risk_scores = np.exp(X @ self.coefficients)
        unique_times = np.sort(np.unique(T[E == 1]))
        baseline = []

        for t in unique_times:
            d_t = np.sum((T == t) & (E == 1))
            at_risk = risk_scores[T >= t].sum()
            baseline.append(d_t / (at_risk + 1e-10))

        self.baseline_hazard = np.array(baseline)
        self.baseline_times = unique_times

    def predict_survival(self, X: np.ndarray) -> np.ndarray:
        """Predict survival probability at mean time."""
        risk_scores = np.exp(X @ self.coefficients)
        cumulative_hazard = np.sum(self.baseline_hazard) if self.baseline_hazard is not None else 0.1
        return np.exp(-cumulative_hazard * risk_scores)

    def concordance_index(self, X, T, E):
        """Compute Harrell's C-index."""
        risk_scores = X @ self.coefficients
        concordant = 0
        discordant = 0
        tied = 0

        event_indices = np.where(E == 1)[0]
        for i in event_indices:
            for j in range(len(T)):
                if T[j] > T[i]:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied += 1

        total = concordant + discordant + tied
        if total == 0:
            return 0.5
        return (concordant + 0.5 * tied) / total


class EADModel:
    """Exposure at Default forecasting framework."""

    def __init__(self, config: dict):
        self.config = config
        self.cox_model = CoxCCFModel()
        self.ccf_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42
        )
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        self._is_fitted = False

    def _get_ead_features(self) -> List[str]:
        """EAD-specific features."""
        return [
            'loan_amount', 'credit_utilization', 'total_credit_limit',
            'revolving_balance', 'undrawn_amount', 'credit_score',
            'dti_ratio', 'interest_rate', 'loan_term_months',
            'employment_years', 'annual_income',
            'purpose_encoded', 'num_credit_lines'
        ]

    def fit(self, df: pd.DataFrame) -> Dict:
        """Train EAD model."""
        logger.info("Training EAD model...")

        available_features = [f for f in self._get_ead_features() if f in df.columns]
        self.feature_names = available_features

        X = df[self.feature_names].copy()
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], 0)

        # Target: actual EAD
        y_ead = df['ead'].values
        y_ccf = df['ccf'].values

        # Time and event for Cox model
        T = np.where(
            df['default_flag'] == 1,
            df['months_to_default'].values,
            df['loan_term_months'].values
        ).astype(float)
        T = np.maximum(T, 1)
        E = df['default_flag'].values

        # Split
        X_train, X_test, y_train, y_test, T_train, T_test, E_train, E_test = \
            train_test_split(X, y_ead, T, E, test_size=0.2, random_state=42)

        # Fit Cox model for survival-based EAD
        logger.info("Fitting Cox regression for EAD...")
        X_train_np = X_train.values.astype(float)
        X_test_np = X_test.values.astype(float)

        # Use subset for Cox (computational efficiency)
        n_cox = min(len(X_train_np), 50000)
        idx = np.random.choice(len(X_train_np), n_cox, replace=False)
        self.cox_model.fit(X_train_np[idx], T_train[idx], E_train[idx])

        # C-index on test set (use subset for speed)
        n_test_cox = min(len(X_test_np), 10000)
        idx_test = np.random.choice(len(X_test_np), n_test_cox, replace=False)
        c_index = self.cox_model.concordance_index(
            X_test_np[idx_test], T_test[idx_test], E_test[idx_test]
        )

        # Fit CCF model
        logger.info("Fitting CCF model...")
        self.ccf_model.fit(X_train, y_train)
        ead_preds = self.ccf_model.predict(X_test)
        ead_preds = np.maximum(ead_preds, X_test['loan_amount'].values * 0.9)

        # Metrics
        mape = mean_absolute_percentage_error(y_test, ead_preds)
        rmse = np.sqrt(mean_squared_error(y_test, ead_preds))

        self.metrics = {
            'c_index': round(c_index, 4),
            'mape': round(mape * 100, 2),
            'rmse': round(rmse, 2),
            'mean_ead_actual': round(np.mean(y_test), 2),
            'mean_ead_predicted': round(np.mean(ead_preds), 2),
            'total_exposure': round(np.sum(y_test), 2)
        }

        self._is_fitted = True
        self._test_actual = y_test
        self._test_predicted = ead_preds

        logger.info(f"EAD Model - C-index: {c_index:.4f}, MAPE: {mape*100:.1f}%")
        return self.metrics

    def predict_ead(self, X: pd.DataFrame) -> np.ndarray:
        """Predict EAD for given features."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_feat = X[self.feature_names].copy()
        X_feat = X_feat.fillna(X_feat.median())
        X_feat = X_feat.replace([np.inf, -np.inf], 0)

        return np.maximum(
            self.ccf_model.predict(X_feat),
            X_feat['loan_amount'].values * 0.9
        )
