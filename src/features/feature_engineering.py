"""
Feature Engineering Module
Implements 100+ credit risk features with WoE/IV analysis
and Kolmogorov-Smirnov validation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class WoETransformer:
    """Weight of Evidence and Information Value calculator."""

    def __init__(self, n_bins: int = 10, min_pct: float = 0.05):
        self.n_bins = n_bins
        self.min_pct = min_pct
        self.woe_maps: Dict[str, Dict] = {}
        self.iv_values: Dict[str, float] = {}

    def _compute_woe_iv(self, feature: pd.Series, target: pd.Series,
                         bins: pd.Series) -> Tuple[Dict, float]:
        """Compute WoE and IV for a binned feature."""
        df = pd.DataFrame({'bin': bins, 'target': target})
        grouped = df.groupby('bin')['target'].agg(['sum', 'count'])
        grouped.columns = ['events', 'total']
        grouped['non_events'] = grouped['total'] - grouped['events']

        total_events = grouped['events'].sum()
        total_non_events = grouped['non_events'].sum()

        # Avoid division by zero
        grouped['event_rate'] = np.maximum(grouped['events'], 0.5) / total_events
        grouped['non_event_rate'] = np.maximum(grouped['non_events'], 0.5) / total_non_events

        grouped['woe'] = np.log(grouped['non_event_rate'] / grouped['event_rate'])
        grouped['iv'] = (grouped['non_event_rate'] - grouped['event_rate']) * grouped['woe']

        woe_map = grouped['woe'].to_dict()
        iv = grouped['iv'].sum()
        return woe_map, iv

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                       features: List[str]) -> pd.DataFrame:
        """Fit WoE transformation and compute IV for all features."""
        result = X.copy()
        logger.info(f"Computing WoE/IV for {len(features)} features...")

        for feat in features:
            try:
                if X[feat].dtype in ['object', 'category']:
                    bins = X[feat].astype(str)
                else:
                    bins = pd.qcut(X[feat], self.n_bins, duplicates='drop',
                                   labels=False).astype(str)

                woe_map, iv = self._compute_woe_iv(X[feat], y, bins)
                self.woe_maps[feat] = woe_map
                self.iv_values[feat] = iv
                result[f'{feat}_woe'] = bins.map(woe_map).fillna(0)
            except Exception as e:
                logger.warning(f"WoE failed for {feat}: {e}")
                self.iv_values[feat] = 0.0

        # Sort by IV
        self.iv_values = dict(
            sorted(self.iv_values.items(), key=lambda x: x[1], reverse=True)
        )
        return result

    def get_iv_summary(self) -> pd.DataFrame:
        """Return IV summary with predictive power categories."""
        df = pd.DataFrame({
            'feature': list(self.iv_values.keys()),
            'iv': list(self.iv_values.values())
        })
        df['predictive_power'] = pd.cut(
            df['iv'],
            bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=['Not useful', 'Weak', 'Medium', 'Strong', 'Very Strong']
        )
        return df


class CreditFeatureEngineer:
    """Engineers 100+ credit risk features from raw loan data."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_stats: Dict[str, Dict] = {}
        self.ks_results: Dict[str, Dict] = {}
        self.woe_transformer = WoETransformer()

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial ratio features."""
        df = df.copy()

        # Income ratios
        df['income_per_credit_line'] = df['annual_income'] / (df['num_credit_lines'] + 1)
        df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['monthly_burden'] = df['monthly_payment'] / (df['monthly_income'] + 1)
        df['total_debt_burden'] = (
            (df['loan_amount'] + df['revolving_balance']) / (df['annual_income'] + 1)
        )

        # Credit ratios
        df['available_credit'] = np.clip(
            df['total_credit_limit'] - df['revolving_balance'], 0, None
        )
        df['available_credit_pct'] = (
            df['available_credit'] / (df['total_credit_limit'] + 1)
        )
        df['credit_depth'] = df['credit_history_months'] / (df['num_credit_lines'] + 1)
        df['delinquency_rate'] = df['num_delinquencies_2y'] / (df['credit_history_months'] / 12 + 1)

        # Property ratios
        df['equity_ratio'] = np.where(
            df['property_value'] > 0,
            1 - df['ltv_ratio'],
            0
        )
        df['property_to_income'] = np.where(
            df['property_value'] > 0,
            df['property_value'] / (df['annual_income'] + 1),
            0
        )

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions capturing non-linear risk patterns."""
        df = df.copy()

        # Risk interactions
        df['score_x_utilization'] = df['credit_score'] * df['credit_utilization']
        df['dti_x_utilization'] = df['dti_ratio'] * df['credit_utilization']
        df['score_x_dti'] = df['credit_score'] * df['dti_ratio']
        df['income_x_term'] = df['annual_income'] * df['loan_term_months']
        df['rate_x_amount'] = df['interest_rate'] * df['loan_amount']
        df['age_x_employment'] = df['age'] * df['employment_years']
        df['delinq_x_utilization'] = df['num_delinquencies_2y'] * df['credit_utilization']
        df['ltv_x_hpi'] = df['ltv_ratio'] * df['hpi_change']

        # Macro interactions
        df['unemployment_x_dti'] = df['unemployment_rate'] * df['dti_ratio']
        df['gdp_x_score'] = df['gdp_growth'] * df['credit_score']
        df['fed_rate_x_amount'] = df['fed_funds_rate'] * df['loan_amount']

        return df

    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial and transformed features."""
        df = df.copy()

        # Squared terms
        df['credit_score_sq'] = df['credit_score'] ** 2
        df['dti_ratio_sq'] = df['dti_ratio'] ** 2
        df['utilization_sq'] = df['credit_utilization'] ** 2
        df['interest_rate_sq'] = df['interest_rate'] ** 2

        # Log transforms
        for col in ['annual_income', 'loan_amount', 'credit_history_months',
                     'total_credit_limit', 'revolving_balance']:
            df[f'{col}_log'] = np.log1p(df[col])

        # Sqrt transforms
        df['employment_sqrt'] = np.sqrt(df['employment_years'])
        df['delinq_sqrt'] = np.sqrt(df['num_delinquencies_2y'])

        return df

    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-grade binned features."""
        df = df.copy()

        # Credit score bins (industry standard)
        df['score_bin'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 620, 660, 700, 740, 780, 850],
            labels=['Deep_Sub', 'Subprime', 'Near_Prime', 'Prime',
                    'Prime_Plus', 'Super_Prime', 'Excellent']
        ).astype(str)

        # DTI bins
        df['dti_bin'] = pd.cut(
            df['dti_ratio'],
            bins=[0, 0.2, 0.35, 0.50, 1.0, 5.0],
            labels=['Low', 'Moderate', 'High', 'Very_High', 'Extreme']
        ).astype(str)

        # LTV bins
        df['ltv_bin'] = pd.cut(
            df['ltv_ratio'],
            bins=[-0.01, 0.0, 0.6, 0.8, 0.95, 1.0, 1.5],
            labels=['No_Collateral', 'Low', 'Moderate', 'High',
                    'Very_High', 'Underwater']
        ).astype(str)

        # Age bins
        df['age_bin'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['Young', 'Early_Career', 'Mid_Career',
                    'Senior', 'Pre_Retire', 'Retired']
        ).astype(str)

        return df

    def create_macro_stress_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create macroeconomic stress indicator features."""
        df = df.copy()

        # Macro stress index
        df['macro_stress_index'] = (
            0.4 * (df['unemployment_rate'] / 10) +
            0.3 * np.clip(-df['gdp_growth'] / 5, 0, 1) +
            0.2 * np.clip(-df['hpi_change'] / 20, 0, 1) +
            0.1 * (df['fed_funds_rate'] / 8)
        ).round(4)

        # Recession indicator
        df['recession_indicator'] = (
            (df['gdp_growth'] < 0) & (df['unemployment_rate'] > 6)
        ).astype(int)

        # Housing stress
        df['housing_stress'] = (
            (df['hpi_change'] < -5) & (df['ltv_ratio'] > 0.8)
        ).astype(int)

        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        cat_cols = ['purpose', 'home_ownership', 'region',
                    'score_bin', 'dti_bin', 'ltv_bin', 'age_bin']

        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        return df

    def validate_features_ks(self, df: pd.DataFrame,
                              target_col: str = 'default_flag') -> pd.DataFrame:
        """Validate features using Kolmogorov-Smirnov test."""
        logger.info("Running KS validation on features...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_col
                        and c not in ['borrower_id', 'pd_true']]

        default = df[df[target_col] == 1]
        non_default = df[df[target_col] == 0]

        results = []
        for col in numeric_cols:
            try:
                ks_stat, p_value = stats.ks_2samp(
                    default[col].dropna(), non_default[col].dropna()
                )
                results.append({
                    'feature': col,
                    'ks_statistic': round(ks_stat, 4),
                    'p_value': p_value,
                    'significant': p_value < 0.01
                })
                self.ks_results[col] = {'ks': ks_stat, 'p': p_value}
            except Exception:
                pass

        results_df = pd.DataFrame(results).sort_values('ks_statistic', ascending=False)
        n_sig = results_df['significant'].sum()
        logger.info(f"KS validation: {n_sig}/{len(results_df)} features significant (p<0.01)")
        return results_df

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline...")

        df = self.create_ratio_features(df)
        df = self.create_interaction_features(df)
        df = self.create_polynomial_features(df)
        df = self.create_binned_features(df)
        df = self.create_macro_stress_features(df)
        df = self.encode_categorical(df)

        n_features = len(df.select_dtypes(include=[np.number]).columns)
        logger.info(f"Feature engineering complete: {n_features} numeric features")

        return df

    def get_model_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of features suitable for modeling."""
        exclude = ['borrower_id', 'default_flag', 'pd_true',
                    'months_to_default', 'lgd', 'recovery_rate',
                    'loss_amount', 'ead', 'origination_date',
                    'origination_quarter']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]
