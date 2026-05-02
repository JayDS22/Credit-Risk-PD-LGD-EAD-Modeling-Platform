"""
Credit Risk Platform - Main Pipeline
Orchestrates the full PD/LGD/EAD modeling workflow.
"""

import sys
import os
import logging
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_generator import CreditDataGenerator
from src.features.feature_engineering import CreditFeatureEngineer, WoETransformer
from src.models.pd_model import PDModel, VasicekCalibrator
from src.models.lgd_model import LGDModel
from src.models.ead_model import EADModel
from src.validation.model_validation import ModelValidator
from src.regulatory.capital_calculation import BaselCapitalCalculator, StressTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / 'configs' / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class CreditRiskPipeline:
    """End-to-end credit risk modeling pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.data: pd.DataFrame = None
        self.feature_engineer = CreditFeatureEngineer()
        self.pd_model = PDModel(config.get('pd_model', {}))
        self.lgd_model = LGDModel(config.get('regulatory', {}))
        self.ead_model = EADModel(config.get('ead_model', {}))
        self.validator = ModelValidator(config.get('validation', {}))
        self.capital_calc = BaselCapitalCalculator(config.get('regulatory', {}))
        self.stress_tester = StressTester(
            config.get('stress_testing', {}).get('scenarios', {})
        )
        self.results = {}

    def step_1_generate_data(self) -> pd.DataFrame:
        """Step 1: Generate synthetic credit risk data."""
        logger.info("=" * 60)
        logger.info("STEP 1: Data Generation")
        logger.info("=" * 60)

        data_config = self.config.get('data', {})
        generator = CreditDataGenerator(
            n_samples=data_config.get('n_samples', 500_000),
            random_seed=data_config.get('random_seed', 42),
            default_rate=data_config.get('default_rate', 0.045)
        )
        self.data = generator.generate_full_dataset()

        self.results['data'] = {
            'n_samples': len(self.data),
            'n_features_raw': self.data.shape[1],
            'default_rate': float(self.data['default_flag'].mean()),
            'n_defaults': int(self.data['default_flag'].sum()),
            'total_exposure': float(self.data['ead'].sum()),
            'mean_loan_amount': float(self.data['loan_amount'].mean()),
        }
        return self.data

    def step_2_engineer_features(self) -> pd.DataFrame:
        """Step 2: Feature engineering."""
        logger.info("=" * 60)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 60)

        self.data = self.feature_engineer.engineer_all_features(self.data)

        # WoE/IV analysis
        model_features = self.feature_engineer.get_model_features(self.data)
        numeric_features = [f for f in model_features
                            if self.data[f].dtype in ['int64', 'float64']]

        woe = WoETransformer(n_bins=10)
        woe.fit_transform(
            self.data[numeric_features[:50]],  # Top 50 for efficiency
            self.data['default_flag'],
            numeric_features[:50]
        )
        iv_summary = woe.get_iv_summary()

        # KS validation
        ks_results = self.feature_engineer.validate_features_ks(
            self.data, 'default_flag'
        )

        self.results['features'] = {
            'n_features_engineered': len(model_features),
            'top_iv_features': iv_summary.head(10).to_dict('records'),
            'n_significant_ks': int(ks_results['significant'].sum()),
            'mean_iv': float(iv_summary['iv'].mean()),
            'max_iv': float(iv_summary['iv'].max()),
        }
        return self.data

    def step_3_train_pd_model(self) -> dict:
        """Step 3: Train PD model."""
        logger.info("=" * 60)
        logger.info("STEP 3: PD Model Training")
        logger.info("=" * 60)

        model_features = self.feature_engineer.get_model_features(self.data)
        pd_results = self.pd_model.fit(
            self.data, self.data['default_flag'], model_features
        )

        # Vasicek calibration
        calibrator = VasicekCalibrator()
        sample_pds = self.pd_model.predict_pd(self.data.head(1000))
        pit_pds = calibrator.ttc_to_pit(sample_pds[:100], macro_factor=0.0)
        stressed_pds = calibrator.stress_pd(sample_pds[:100], stress_factor=-2.0)

        self.results['pd_model'] = {
            'selected_model': pd_results['selected'],
            'auc': pd_results['metrics']['auc'],
            'gini': pd_results['metrics']['gini'],
            'ks_statistic': pd_results['metrics']['ks_statistic'],
            'logistic_auc': pd_results['logistic']['auc'],
            'xgboost_auc': pd_results['xgboost']['auc'],
            'mean_pit_pd': float(np.mean(pit_pds)),
            'mean_stressed_pd': float(np.mean(stressed_pds)),
            'risk_grade_stats': self.pd_model.risk_grade_stats.to_dict('records')
                if hasattr(self.pd_model, 'risk_grade_stats') else []
        }

        return pd_results

    def step_4_train_lgd_model(self) -> dict:
        """Step 4: Train LGD model."""
        logger.info("=" * 60)
        logger.info("STEP 4: LGD Model Training")
        logger.info("=" * 60)

        lgd_results = self.lgd_model.fit(self.data)

        self.results['lgd_model'] = lgd_results
        return lgd_results

    def step_5_train_ead_model(self) -> dict:
        """Step 5: Train EAD model."""
        logger.info("=" * 60)
        logger.info("STEP 5: EAD Model Training")
        logger.info("=" * 60)

        ead_results = self.ead_model.fit(self.data)

        self.results['ead_model'] = ead_results
        return ead_results

    def step_6_validate_models(self) -> dict:
        """Step 6: Model validation."""
        logger.info("=" * 60)
        logger.info("STEP 6: Model Validation")
        logger.info("=" * 60)

        y_true = self.pd_model._test_actual
        y_prob = self.pd_model._test_probs

        validation = self.validator.run_full_validation(
            y_true.values if hasattr(y_true, 'values') else y_true,
            y_prob
        )

        self.results['validation'] = {
            'hosmer_lemeshow': {
                'chi_squared': validation['hosmer_lemeshow']['chi_squared'],
                'p_value': validation['hosmer_lemeshow']['p_value'],
                'pass': validation['hosmer_lemeshow']['pass']
            },
            'psi': {
                'value': validation['psi']['psi'],
                'stability': validation['psi']['stability']
            },
            'binomial_backtest': {
                'ae_ratio': validation['binomial_backtest']['actual_to_expected_ratio'],
                'zone': validation['binomial_backtest']['traffic_light_zone'],
                'pass': validation['binomial_backtest']['pass']
            },
            'discrimination': {
                'auc': validation['discrimination']['auc'],
                'gini': validation['discrimination']['gini'],
                'ks': validation['discrimination']['ks_statistic'],
                'accuracy_ratio': validation['discrimination']['accuracy_ratio']
            }
        }
        return validation

    def step_7_regulatory_capital(self) -> dict:
        """Step 7: Regulatory capital calculation."""
        logger.info("=" * 60)
        logger.info("STEP 7: Regulatory Capital & Stress Testing")
        logger.info("=" * 60)

        # Use model predictions for capital calculation
        sample = self.data.sample(min(100_000, len(self.data)), random_state=42)
        pd_pred = self.pd_model.predict_pd(sample)

        # For defaulted accounts, use LGD model; for non-defaulted, use regulatory floor
        lgd_pred = np.where(
            sample['default_flag'] == 1,
            sample['lgd'].values,
            np.clip(0.45 + np.random.normal(0, 0.05, len(sample)), 0.10, 0.90)
        )
        ead_pred = sample['ead'].values

        # Expected Loss
        el_result = self.capital_calc.compute_expected_loss(pd_pred, lgd_pred, ead_pred)

        # RWA
        rwa_result = self.capital_calc.compute_rwa(
            pd_pred, lgd_pred, ead_pred, approach='irb_advanced'
        )

        # Economic Capital
        ec_result = self.capital_calc.economic_capital(el_result['el_individual'])

        # Stress Testing
        stress_results = self.stress_tester.run_scenarios(pd_pred, lgd_pred, ead_pred)

        self.results['regulatory'] = {
            'expected_loss': {
                'total': el_result['el_total'],
                'rate': el_result['el_rate'],
                'total_exposure': el_result['total_exposure'],
            },
            'rwa': {
                'total': rwa_result['rwa_total'],
                'density': rwa_result['rwa_density'],
                'avg_risk_weight': rwa_result['avg_risk_weight'],
                'capital_required': rwa_result['total_capital_required'],
            },
            'economic_capital': {
                'ec': ec_result['economic_capital'],
                'var': ec_result['var'],
                'expected_shortfall': ec_result['expected_shortfall'],
            },
            'stress_test': {
                name: {
                    'mean_pd': r['mean_pd'],
                    'total_el': r['total_el'],
                    'pd_migration_pct': r['pd_migration_pct']
                }
                for name, r in stress_results.items()
            }
        }
        return self.results['regulatory']

    def run_full_pipeline(self) -> dict:
        """Execute the complete credit risk modeling pipeline."""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("CREDIT RISK PD/LGD/EAD MODELING PLATFORM")
        logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        self.step_1_generate_data()
        self.step_2_engineer_features()
        self.step_3_train_pd_model()
        self.step_4_train_lgd_model()
        self.step_5_train_ead_model()
        self.step_6_validate_models()
        self.step_7_regulatory_capital()

        elapsed = (datetime.now() - start_time).total_seconds()
        self.results['pipeline'] = {
            'execution_time_seconds': round(elapsed, 1),
            'timestamp': start_time.isoformat(),
            'status': 'completed'
        }

        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 60)

        return self.results

    def export_results(self, output_path: str = 'results.json'):
        """Export pipeline results to JSON."""
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj

        clean_results = json.loads(
            json.dumps(self.results, default=convert)
        )

        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)

        logger.info(f"Results exported to {output_path}")
        return clean_results


def main():
    """Main entry point."""
    config = load_config()
    pipeline = CreditRiskPipeline(config)
    results = pipeline.run_full_pipeline()
    pipeline.export_results('results.json')

    # Print key metrics
    print("\n" + "=" * 60)
    print("KEY RESULTS SUMMARY")
    print("=" * 60)

    if 'pd_model' in results:
        pd_r = results['pd_model']
        print(f"\nPD Model ({pd_r['selected_model']}):")
        print(f"  AUC: {pd_r['auc']}")
        print(f"  Gini: {pd_r['gini']}")
        print(f"  KS Statistic: {pd_r['ks_statistic']}")

    if 'lgd_model' in results:
        lgd_r = results['lgd_model']
        print(f"\nLGD Model:")
        print(f"  R2: {lgd_r.get('r2', 'N/A')}")
        print(f"  RMSE: {lgd_r.get('rmse', 'N/A')}")

    if 'ead_model' in results:
        ead_r = results['ead_model']
        print(f"\nEAD Model:")
        print(f"  C-index: {ead_r.get('c_index', 'N/A')}")
        print(f"  MAPE: {ead_r.get('mape', 'N/A')}%")

    if 'validation' in results:
        val_r = results['validation']
        print(f"\nValidation:")
        print(f"  H-L Test: chi2={val_r['hosmer_lemeshow']['chi_squared']}, "
              f"p={val_r['hosmer_lemeshow']['p_value']}")
        print(f"  PSI: {val_r['psi']['value']} ({val_r['psi']['stability']})")
        print(f"  Backtest A/E: {val_r['binomial_backtest']['ae_ratio']} "
              f"({val_r['binomial_backtest']['zone']})")

    if 'regulatory' in results:
        reg_r = results['regulatory']
        print(f"\nRegulatory Capital:")
        print(f"  Total EL: ${reg_r['expected_loss']['total']:,.0f}")
        print(f"  Total RWA: ${reg_r['rwa']['total']:,.0f}")
        print(f"  Capital Required: ${reg_r['rwa']['capital_required']:,.0f}")

    return results


if __name__ == '__main__':
    main()
