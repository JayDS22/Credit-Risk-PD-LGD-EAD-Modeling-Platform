"""
Model Validation Framework
Hosmer-Lemeshow, PSI, back-testing, and traffic light approach.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """Comprehensive credit risk model validation suite."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.validation_results: Dict = {}

    def hosmer_lemeshow_test(self, y_true: np.ndarray, y_prob: np.ndarray,
                              n_groups: int = 10) -> Dict:
        """Hosmer-Lemeshow goodness-of-fit test."""
        df = pd.DataFrame({'actual': y_true, 'predicted': y_prob})
        df['group'] = pd.qcut(df['predicted'], n_groups, duplicates='drop',
                               labels=False)

        grouped = df.groupby('group').agg(
            n=('actual', 'size'),
            observed=('actual', 'sum'),
            expected_prob=('predicted', 'mean')
        )
        grouped['expected'] = grouped['n'] * grouped['expected_prob']

        # Chi-squared statistic
        chi2 = np.sum(
            (grouped['observed'] - grouped['expected']) ** 2 /
            (grouped['expected'] * (1 - grouped['expected_prob']) + 1e-10)
        )

        dof = len(grouped) - 2
        p_value = 1 - stats.chi2.cdf(chi2, dof)

        result = {
            'chi_squared': round(chi2, 2),
            'p_value': round(p_value, 4),
            'degrees_of_freedom': dof,
            'pass': p_value > 0.05,
            'interpretation': (
                'Model fits well (fail to reject H0)'
                if p_value > 0.05
                else 'Poor fit (reject H0)'
            ),
            'group_details': grouped.to_dict()
        }

        self.validation_results['hosmer_lemeshow'] = result
        logger.info(f"Hosmer-Lemeshow: chi2={chi2:.2f}, p={p_value:.4f}")
        return result

    def population_stability_index(self, expected: np.ndarray,
                                     actual: np.ndarray,
                                     n_bins: int = 10) -> Dict:
        """Population Stability Index (PSI) for distribution shift detection."""
        # Create bins from expected distribution
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]

        # Proportions (with smoothing)
        expected_pct = (expected_counts + 0.5) / (len(expected) + n_bins * 0.5)
        actual_pct = (actual_counts + 0.5) / (len(actual) + n_bins * 0.5)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        if psi < 0.10:
            interpretation = 'No significant shift (Green)'
            stability = 'stable'
        elif psi < 0.25:
            interpretation = 'Moderate shift - investigate (Yellow)'
            stability = 'warning'
        else:
            interpretation = 'Significant shift - action required (Red)'
            stability = 'unstable'

        result = {
            'psi': round(psi, 4),
            'interpretation': interpretation,
            'stability': stability,
            'n_bins': n_bins,
            'bin_details': {
                'expected_pct': expected_pct.tolist(),
                'actual_pct': actual_pct.tolist()
            }
        }

        self.validation_results['psi'] = result
        logger.info(f"PSI: {psi:.4f} - {stability}")
        return result

    def binomial_backtest(self, n_defaults_actual: int, n_total: int,
                           pd_predicted: float,
                           confidence: float = 0.95) -> Dict:
        """Binomial test for back-testing PD predictions."""
        expected_defaults = n_total * pd_predicted
        actual_to_expected = n_defaults_actual / max(expected_defaults, 1)

        # Binomial test: is actual significantly different from expected?
        p_value = stats.binom_test(
            n_defaults_actual, n_total, pd_predicted, alternative='two-sided'
        ) if hasattr(stats, 'binom_test') else stats.binomtest(
            n_defaults_actual, n_total, pd_predicted, alternative='two-sided'
        ).pvalue

        # Traffic light approach
        green_threshold = stats.binom.ppf(0.95, n_total, pd_predicted)
        yellow_threshold = stats.binom.ppf(0.9999, n_total, pd_predicted)

        if n_defaults_actual <= green_threshold:
            zone = 'green'
        elif n_defaults_actual <= yellow_threshold:
            zone = 'yellow'
        else:
            zone = 'red'

        result = {
            'n_defaults_actual': n_defaults_actual,
            'n_defaults_expected': round(expected_defaults, 1),
            'n_total': n_total,
            'pd_predicted': round(pd_predicted, 6),
            'actual_default_rate': round(n_defaults_actual / n_total, 6),
            'actual_to_expected_ratio': round(actual_to_expected, 4),
            'p_value': round(p_value, 6),
            'traffic_light_zone': zone,
            'green_threshold': int(green_threshold),
            'yellow_threshold': int(yellow_threshold),
            'pass': p_value > 0.05
        }

        self.validation_results['binomial_backtest'] = result
        logger.info(f"Backtest: A/E ratio={actual_to_expected:.4f}, "
                     f"Zone={zone}, p={p_value:.4f}")
        return result

    def discrimination_metrics(self, y_true: np.ndarray,
                                y_prob: np.ndarray) -> Dict:
        """Comprehensive discrimination metrics."""
        from sklearn.metrics import roc_auc_score, roc_curve

        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1

        # KS statistic
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks = np.max(tpr - fpr)

        # Accuracy Ratio / CAP curve
        sorted_idx = np.argsort(-y_prob)
        y_sorted = y_true[sorted_idx] if isinstance(y_true, np.ndarray) else y_true.values[sorted_idx]
        cum_defaults = np.cumsum(y_sorted) / np.sum(y_sorted)
        cum_total = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

        # AR = area under CAP / area under perfect model
        ar_model = np.trapezoid(cum_defaults, cum_total)
        default_rate = np.mean(y_true)
        ar_perfect = 1 - default_rate / 2
        ar_random = 0.5
        accuracy_ratio = (ar_model - ar_random) / (ar_perfect - ar_random)

        result = {
            'auc': round(auc, 4),
            'gini': round(gini, 4),
            'ks_statistic': round(ks, 4),
            'accuracy_ratio': round(accuracy_ratio, 4),
            'cap_curve': {
                'cum_defaults': cum_defaults[::max(1, len(cum_defaults)//100)].tolist(),
                'cum_total': cum_total[::max(1, len(cum_total)//100)].tolist()
            }
        }

        self.validation_results['discrimination'] = result
        return result

    def migration_matrix(self, grades_t0: np.ndarray,
                          grades_t1: np.ndarray,
                          grade_labels: List[str] = None) -> pd.DataFrame:
        """Compute rating migration matrix."""
        if grade_labels is None:
            all_grades = sorted(set(grades_t0) | set(grades_t1))
            grade_labels = [str(g) for g in all_grades]

        migration = pd.crosstab(
            pd.Categorical(grades_t0, categories=grade_labels),
            pd.Categorical(grades_t1, categories=grade_labels),
            normalize='index'
        )
        migration.index.name = 'From'
        migration.columns.name = 'To'

        self.validation_results['migration_matrix'] = migration
        return migration

    def run_full_validation(self, y_true: np.ndarray, y_prob: np.ndarray,
                             expected_dist: np.ndarray = None) -> Dict:
        """Run complete validation suite."""
        logger.info("Running full model validation suite...")

        results = {}

        # Discrimination
        results['discrimination'] = self.discrimination_metrics(y_true, y_prob)

        # Hosmer-Lemeshow
        results['hosmer_lemeshow'] = self.hosmer_lemeshow_test(y_true, y_prob)

        # PSI (compare predicted distributions)
        if expected_dist is not None:
            results['psi'] = self.population_stability_index(expected_dist, y_prob)
        else:
            # Use train/test split as proxy
            mid = len(y_prob) // 2
            results['psi'] = self.population_stability_index(
                y_prob[:mid], y_prob[mid:]
            )

        # Binomial backtest
        n_defaults = int(np.sum(y_true))
        n_total = len(y_true)
        mean_pd = np.mean(y_prob)
        results['binomial_backtest'] = self.binomial_backtest(
            n_defaults, n_total, mean_pd
        )

        self.validation_results = results
        return results

    def get_validation_summary(self) -> pd.DataFrame:
        """Generate validation summary report."""
        rows = []
        for test_name, result in self.validation_results.items():
            if isinstance(result, dict):
                for metric, value in result.items():
                    if isinstance(value, (int, float, str, bool)):
                        rows.append({
                            'test': test_name,
                            'metric': metric,
                            'value': value
                        })

        return pd.DataFrame(rows)
