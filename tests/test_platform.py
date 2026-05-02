"""
Test Suite for Credit Risk Platform
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_generator import CreditDataGenerator
from src.features.feature_engineering import CreditFeatureEngineer, WoETransformer
from src.models.pd_model import PDModel, VasicekCalibrator
from src.models.lgd_model import LGDModel
from src.models.ead_model import EADModel
from src.validation.model_validation import ModelValidator
from src.regulatory.capital_calculation import BaselCapitalCalculator, StressTester


# Use small dataset for testing
N_SAMPLES = 5000
DEFAULT_RATE = 0.05


@pytest.fixture(scope='module')
def sample_data():
    """Generate small sample dataset for testing."""
    gen = CreditDataGenerator(n_samples=N_SAMPLES, random_seed=42,
                               default_rate=DEFAULT_RATE)
    return gen.generate_full_dataset()


@pytest.fixture(scope='module')
def engineered_data(sample_data):
    """Feature-engineered dataset."""
    fe = CreditFeatureEngineer()
    return fe.engineer_all_features(sample_data), fe


class TestDataGenerator:
    def test_generates_correct_size(self, sample_data):
        assert len(sample_data) == N_SAMPLES

    def test_has_required_columns(self, sample_data):
        required = ['borrower_id', 'credit_score', 'loan_amount',
                     'default_flag', 'lgd', 'ead', 'pd_true']
        for col in required:
            assert col in sample_data.columns, f"Missing column: {col}"

    def test_default_rate_reasonable(self, sample_data):
        dr = sample_data['default_flag'].mean()
        assert 0.01 < dr < 0.15, f"Default rate {dr} out of range"

    def test_credit_score_range(self, sample_data):
        assert sample_data['credit_score'].min() >= 300
        assert sample_data['credit_score'].max() <= 850

    def test_lgd_bounded(self, sample_data):
        defaults = sample_data[sample_data['default_flag'] == 1]
        if len(defaults) > 0:
            assert defaults['lgd'].min() >= 0
            assert defaults['lgd'].max() <= 1

    def test_ead_positive(self, sample_data):
        assert (sample_data['ead'] > 0).all()


class TestFeatureEngineering:
    def test_creates_features(self, engineered_data):
        df, fe = engineered_data
        features = fe.get_model_features(df)
        assert len(features) > 30, f"Expected >30 features, got {len(features)}"

    def test_no_null_in_features(self, engineered_data):
        df, fe = engineered_data
        features = fe.get_model_features(df)
        numeric_df = df[features].select_dtypes(include=[np.number])
        null_count = numeric_df.isnull().sum().sum()
        inf_count = np.isinf(numeric_df.values).sum()
        # Allow some but flag excessive
        assert null_count < len(df) * len(features) * 0.01

    def test_ks_validation(self, engineered_data):
        df, fe = engineered_data
        ks_results = fe.validate_features_ks(df)
        assert len(ks_results) > 0
        assert 'ks_statistic' in ks_results.columns

    def test_woe_transformer(self, engineered_data):
        df, fe = engineered_data
        woe = WoETransformer(n_bins=5)
        features = ['credit_score', 'dti_ratio', 'credit_utilization']
        result = woe.fit_transform(df[features], df['default_flag'], features)
        iv_summary = woe.get_iv_summary()
        assert len(iv_summary) == len(features)
        assert all(iv_summary['iv'] >= 0)


class TestPDModel:
    def test_pd_model_trains(self, engineered_data):
        df, fe = engineered_data
        config = {
            'xgboost_params': {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.1,
                'scale_pos_weight': 10
            }
        }
        model = PDModel(config)
        features = fe.get_model_features(df)
        results = model.fit(df, df['default_flag'], features)

        assert results['metrics']['auc'] > 0.5
        assert results['metrics']['gini'] > 0.0
        assert results['metrics']['ks_statistic'] > 0.0

    def test_pd_predictions_bounded(self, engineered_data):
        df, fe = engineered_data
        config = {'xgboost_params': {'n_estimators': 20, 'max_depth': 3,
                                      'scale_pos_weight': 10}}
        model = PDModel(config)
        features = fe.get_model_features(df)
        model.fit(df, df['default_flag'], features)
        preds = model.predict_pd(df.head(100))
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_vasicek_calibrator(self):
        cal = VasicekCalibrator()
        pd_ttc = np.array([0.01, 0.05, 0.10, 0.20])
        pit = cal.ttc_to_pit(pd_ttc, macro_factor=0.0)
        assert len(pit) == len(pd_ttc)
        assert np.all(pit >= 0) and np.all(pit <= 1)

        # Stress should increase PD
        stressed = cal.stress_pd(pd_ttc, stress_factor=-2.0)
        assert np.all(stressed >= pd_ttc)


class TestLGDModel:
    def test_lgd_model_trains(self, engineered_data):
        df, fe = engineered_data
        model = LGDModel({})
        results = model.fit(df)
        if results:
            assert results['r2'] > -1  # Can be negative for poor fits
            assert results['rmse'] > 0
            assert results['rmse'] < 1


class TestEADModel:
    def test_ead_model_trains(self, engineered_data):
        df, fe = engineered_data
        model = EADModel({})
        results = model.fit(df)
        assert results['mape'] > 0
        assert results['c_index'] > 0


class TestModelValidation:
    def test_hosmer_lemeshow(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.1, 1000)
        y_prob = np.clip(y_true * 0.3 + np.random.normal(0.1, 0.05, 1000), 0.01, 0.99)
        validator = ModelValidator()
        result = validator.hosmer_lemeshow_test(y_true, y_prob)
        assert 'chi_squared' in result
        assert 'p_value' in result

    def test_psi(self):
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.1, 1, 1000)
        validator = ModelValidator()
        result = validator.population_stability_index(expected, actual)
        assert result['psi'] >= 0

    def test_binomial_backtest(self):
        validator = ModelValidator()
        result = validator.binomial_backtest(50, 1000, 0.05)
        assert result['traffic_light_zone'] in ['green', 'yellow', 'red']


class TestRegulatoryCapital:
    def test_expected_loss(self):
        calc = BaselCapitalCalculator()
        pd = np.array([0.01, 0.05, 0.10])
        lgd = np.array([0.45, 0.45, 0.45])
        ead = np.array([100000, 200000, 50000])
        result = calc.compute_expected_loss(pd, lgd, ead)
        assert result['el_total'] > 0
        assert result['el_total'] == pytest.approx(
            0.01*0.45*100000 + 0.05*0.45*200000 + 0.10*0.45*50000,
            rel=1e-6
        )

    def test_irb_risk_weight(self):
        calc = BaselCapitalCalculator()
        rw = calc.irb_risk_weight(pd=0.01, lgd=0.45)
        assert rw > 0
        # Higher PD should give higher risk weight
        rw_high = calc.irb_risk_weight(pd=0.10, lgd=0.45)
        assert rw_high > rw

    def test_stress_testing(self):
        tester = StressTester()
        pd_base = np.array([0.01, 0.02, 0.05, 0.10])
        lgd_base = np.array([0.40, 0.45, 0.50, 0.55])
        ead = np.array([100000, 200000, 150000, 50000])
        results = tester.run_scenarios(pd_base, lgd_base, ead)
        assert 'baseline' in results
        assert 'adverse' in results
        # Adverse should have higher EL
        assert results['adverse']['total_el'] >= results['baseline']['total_el']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
