"""
Synthetic Credit Risk Data Generator
Generates realistic loan portfolio data for PD/LGD/EAD modeling
with Basel III-compliant features and performance outcomes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CreditDataGenerator:
    """Generates synthetic credit risk data with realistic distributions."""

    def __init__(self, n_samples: int = 500_000, random_seed: int = 42,
                 default_rate: float = 0.045):
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.default_rate = default_rate
        np.random.seed(random_seed)

    def generate_borrower_features(self) -> pd.DataFrame:
        """Generate borrower demographic and financial features."""
        n = self.n_samples
        logger.info(f"Generating {n:,} borrower profiles...")

        # Demographics
        age = np.clip(np.random.lognormal(3.6, 0.25, n), 18, 85).astype(int)
        employment_years = np.clip(
            np.random.exponential(5, n) + np.random.normal(0, 1, n), 0, 45
        ).round(1)
        annual_income = np.clip(
            np.random.lognormal(10.8, 0.7, n), 15000, 2_000_000
        ).round(0)
        monthly_income = annual_income / 12

        # Credit history
        credit_score = np.clip(
            np.random.normal(680, 80, n) + (employment_years * 2), 300, 850
        ).astype(int)
        credit_history_months = np.clip(
            np.random.gamma(5, 20, n) + age * 2, 6, 600
        ).astype(int)
        num_credit_lines = np.clip(
            np.random.poisson(5, n) + np.random.randint(0, 3, n), 1, 30
        )
        num_delinquencies_2y = np.random.poisson(0.3, n)
        num_delinquencies_2y = np.clip(num_delinquencies_2y, 0, 10)

        # Loan characteristics
        loan_amount = np.clip(
            np.random.lognormal(10.0, 0.8, n), 1000, 500_000
        ).round(0)
        loan_term_months = np.random.choice(
            [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360],
            n, p=[0.02, 0.05, 0.15, 0.10, 0.15, 0.08, 0.05,
                  0.08, 0.07, 0.10, 0.15]
        )
        interest_rate = np.clip(
            12 - (credit_score - 300) / 55 + np.random.normal(0, 1.5, n),
            2.5, 28.0
        ).round(2)

        # Financial ratios
        monthly_payment = (
            loan_amount * (interest_rate / 1200) /
            (1 - (1 + interest_rate / 1200) ** (-loan_term_months))
        ).round(2)
        total_debt = loan_amount + np.clip(
            np.random.lognormal(9.5, 1.0, n), 0, 1_000_000
        )
        dti_ratio = np.clip(total_debt / (annual_income + 1), 0.01, 5.0).round(4)
        payment_to_income = np.clip(
            monthly_payment / (monthly_income + 1), 0.01, 1.5
        ).round(4)

        # Credit utilization
        total_credit_limit = np.clip(
            annual_income * np.random.uniform(0.3, 2.0, n), 5000, 500_000
        ).round(0)
        credit_utilization = np.clip(
            np.random.beta(2, 5, n) + (num_delinquencies_2y * 0.05), 0, 1.0
        ).round(4)
        revolving_balance = (total_credit_limit * credit_utilization).round(0)

        # Property and collateral
        property_value = np.where(
            loan_term_months >= 120,
            np.clip(np.random.lognormal(12.0, 0.5, n), 50000, 2_000_000),
            0
        ).round(0)
        ltv_ratio = np.where(
            property_value > 0,
            np.clip(loan_amount / (property_value + 1), 0.1, 1.5),
            0
        ).round(4)

        # Macroeconomic indicators (at origination)
        unemployment_rate = np.clip(
            np.random.normal(5.5, 1.5, n), 2.0, 15.0
        ).round(1)
        gdp_growth = np.clip(
            np.random.normal(2.5, 1.2, n), -5.0, 8.0
        ).round(1)
        fed_funds_rate = np.clip(
            np.random.normal(3.0, 1.5, n), 0.0, 8.0
        ).round(2)
        hpi_change = np.clip(
            np.random.normal(3.0, 4.0, n), -20.0, 25.0
        ).round(1)

        # Categorical features
        purpose = np.random.choice(
            ['mortgage', 'auto', 'personal', 'credit_card',
             'small_business', 'student', 'home_equity'],
            n, p=[0.25, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10]
        )
        home_ownership = np.random.choice(
            ['own', 'mortgage', 'rent', 'other'],
            n, p=[0.20, 0.35, 0.40, 0.05]
        )
        region = np.random.choice(
            ['northeast', 'southeast', 'midwest', 'southwest', 'west'],
            n, p=[0.22, 0.20, 0.18, 0.18, 0.22]
        )

        df = pd.DataFrame({
            'borrower_id': np.arange(1, n + 1),
            'age': age,
            'employment_years': employment_years,
            'annual_income': annual_income,
            'monthly_income': monthly_income,
            'credit_score': credit_score,
            'credit_history_months': credit_history_months,
            'num_credit_lines': num_credit_lines,
            'num_delinquencies_2y': num_delinquencies_2y,
            'loan_amount': loan_amount,
            'loan_term_months': loan_term_months,
            'interest_rate': interest_rate,
            'monthly_payment': monthly_payment,
            'dti_ratio': dti_ratio,
            'payment_to_income': payment_to_income,
            'total_credit_limit': total_credit_limit,
            'credit_utilization': credit_utilization,
            'revolving_balance': revolving_balance,
            'property_value': property_value,
            'ltv_ratio': ltv_ratio,
            'unemployment_rate': unemployment_rate,
            'gdp_growth': gdp_growth,
            'fed_funds_rate': fed_funds_rate,
            'hpi_change': hpi_change,
            'purpose': purpose,
            'home_ownership': home_ownership,
            'region': region,
        })
        return df

    def generate_default_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate default outcomes based on risk factors."""
        logger.info("Generating default outcomes...")

        # Default probability logit model
        logit = (
            -3.5
            - 0.03 * (df['credit_score'] - 600)
            + 0.8 * df['dti_ratio']
            + 1.2 * df['credit_utilization']
            + 0.15 * df['num_delinquencies_2y']
            - 0.02 * df['employment_years']
            + 0.5 * df['payment_to_income']
            + 0.3 * np.where(df['ltv_ratio'] > 0.8, df['ltv_ratio'] - 0.8, 0)
            + 0.1 * df['unemployment_rate']
            - 0.05 * df['gdp_growth']
            + np.random.normal(0, 0.3, len(df))
        )

        pd_true = 1 / (1 + np.exp(-logit))
        # Calibrate to target default rate
        threshold = np.percentile(pd_true, (1 - self.default_rate) * 100)
        df['default_flag'] = (pd_true >= threshold).astype(int)
        df['pd_true'] = pd_true.round(6)

        # Months to default (for defaulted accounts)
        default_mask = df['default_flag'] == 1
        n_defaults = default_mask.sum()
        df['months_to_default'] = 0
        df.loc[default_mask, 'months_to_default'] = np.clip(
            np.random.weibull(1.5, n_defaults) * 8 + 3, 1, 12
        ).astype(int)

        logger.info(f"Default rate: {df['default_flag'].mean():.4f} "
                     f"({n_defaults:,} defaults out of {len(df):,})")
        return df

    def generate_lgd_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate loss given default for defaulted accounts."""
        logger.info("Generating LGD outcomes...")
        default_mask = df['default_flag'] == 1

        # LGD depends on collateral, seniority, and macro conditions
        lgd_base = np.where(
            df['property_value'] > 0,
            np.clip(0.25 + 0.15 * df['ltv_ratio'] - 0.05 * df['hpi_change'] / 100, 0.05, 0.95),
            np.clip(0.45 + 0.1 * df['credit_utilization'] + 0.02 * df['unemployment_rate'], 0.1, 0.98)
        )
        # Add noise from beta distribution
        alpha = lgd_base * 5
        beta_param = (1 - lgd_base) * 5
        lgd_realized = np.clip(
            np.random.beta(np.maximum(alpha, 0.1), np.maximum(beta_param, 0.1)),
            0.01, 0.99
        )

        df['lgd'] = np.where(default_mask, lgd_realized.round(4), 0.0)
        df['recovery_rate'] = np.where(default_mask, (1 - df['lgd']).round(4), 0.0)
        df['loss_amount'] = np.where(
            default_mask,
            (df['loan_amount'] * df['lgd']).round(2),
            0.0
        )

        if default_mask.sum() > 0:
            logger.info(f"Mean LGD (defaulted): {df.loc[default_mask, 'lgd'].mean():.4f}")
        return df

    def generate_ead_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate exposure at default with credit conversion factors."""
        logger.info("Generating EAD outcomes...")

        # Committed but undrawn amounts for revolving facilities
        is_revolving = df['purpose'].isin(['credit_card', 'home_equity'])
        undrawn_amount = np.where(
            is_revolving,
            np.clip(df['total_credit_limit'] - df['revolving_balance'], 0, None),
            0
        )

        # Credit conversion factor
        ccf = np.where(
            is_revolving,
            np.clip(
                0.5 + 0.3 * df['credit_utilization'] + np.random.normal(0, 0.1, len(df)),
                0.1, 1.0
            ),
            1.0
        )

        df['undrawn_amount'] = undrawn_amount.round(0)
        df['ccf'] = ccf.round(4)
        df['ead'] = np.clip(
            df['loan_amount'] + df['undrawn_amount'] * df['ccf'],
            df['loan_amount'] * 0.9,
            df['loan_amount'] * 1.5
        ).round(2)

        logger.info(f"Total portfolio EAD: ${df['ead'].sum():,.0f}")
        return df

    def generate_full_dataset(self) -> pd.DataFrame:
        """Generate complete credit risk dataset."""
        df = self.generate_borrower_features()
        df = self.generate_default_outcomes(df)
        df = self.generate_lgd_outcomes(df)
        df = self.generate_ead_outcomes(df)

        # Add origination dates for vintage analysis
        df['origination_date'] = pd.date_range(
            start='2020-01-01', periods=self.n_samples, freq='min'
        )[:self.n_samples]
        df['origination_quarter'] = df['origination_date'].dt.to_period('Q').astype(str)

        logger.info(f"Dataset generated: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df


def load_or_generate_data(config: dict, cache_path: Optional[str] = None) -> pd.DataFrame:
    """Load cached data or generate new synthetic dataset."""
    if cache_path:
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded cached data from {cache_path}")
            return df
        except FileNotFoundError:
            pass

    gen = CreditDataGenerator(
        n_samples=config.get('n_samples', 500_000),
        random_seed=config.get('random_seed', 42),
        default_rate=config.get('default_rate', 0.045)
    )
    df = gen.generate_full_dataset()

    if cache_path:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached data to {cache_path}")

    return df
