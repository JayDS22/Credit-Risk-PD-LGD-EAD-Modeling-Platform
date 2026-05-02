"""
Regulatory Capital Calculation
Basel III Standardized and IRB Advanced approaches
with Expected Loss and Economic Capital.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaselCapitalCalculator:
    """Basel III regulatory capital calculator."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.framework = self.config.get('framework', 'basel_iii')

    def compute_expected_loss(self, pd: np.ndarray, lgd: np.ndarray,
                               ead: np.ndarray) -> Dict:
        """Compute Expected Loss = PD x LGD x EAD."""
        el = pd * lgd * ead
        total_exposure = np.sum(ead)

        result = {
            'el_individual': el,
            'el_total': float(np.sum(el)),
            'el_mean': float(np.mean(el)),
            'el_median': float(np.median(el)),
            'el_std': float(np.std(el)),
            'el_rate': float(np.sum(el) / total_exposure) if total_exposure > 0 else 0,
            'total_exposure': float(total_exposure),
            'n_accounts': len(pd),
            'confidence_intervals': {
                '95': float(np.percentile(el, 95)),
                '99': float(np.percentile(el, 99)),
                '99.9': float(np.percentile(el, 99.9)),
            }
        }

        logger.info(f"Expected Loss: ${result['el_total']:,.0f} "
                     f"({result['el_rate']*100:.2f}% of exposure)")
        return result

    def irb_risk_weight(self, pd: float, lgd: float,
                         maturity: float = 2.5) -> float:
        """Compute IRB risk weight for a single exposure.

        Based on Basel III IRB formula for corporate/retail exposures.
        """
        pd = max(pd, 0.0003)  # Floor at 3bps

        # Asset correlation (Basel formula)
        rho = (
            0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) +
            0.24 * (1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)))
        )

        # Maturity adjustment
        b = (0.11852 - 0.05478 * np.log(pd)) ** 2
        ma = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

        # Conditional PD at 99.9% confidence
        z_999 = stats.norm.ppf(0.999)
        conditional_pd = stats.norm.cdf(
            (stats.norm.ppf(pd) + np.sqrt(rho) * z_999) / np.sqrt(1 - rho)
        )

        # Capital requirement (K)
        k = (lgd * conditional_pd - pd * lgd) * ma
        k = max(k, 0)

        # Risk weight
        rw = k * 12.5
        return rw

    def compute_rwa(self, pd: np.ndarray, lgd: np.ndarray,
                     ead: np.ndarray, maturity: float = 2.5,
                     approach: str = 'irb_advanced') -> Dict:
        """Compute Risk-Weighted Assets."""
        if approach == 'irb_advanced':
            risk_weights = np.array([
                self.irb_risk_weight(p, l, maturity)
                for p, l in zip(pd, lgd)
            ])
            rwa = ead * risk_weights
        else:
            # Standardized approach - simplified risk weights
            risk_weights = np.where(
                pd < 0.01, 0.20,
                np.where(pd < 0.03, 0.50,
                         np.where(pd < 0.10, 1.00,
                                  np.where(pd < 0.20, 1.50, 2.50)))
            )
            rwa = ead * risk_weights

        total_rwa = float(np.sum(rwa))
        total_exposure = float(np.sum(ead))

        # Capital requirements
        min_capital_ratio = 0.08  # 8% minimum
        capital_conservation = 0.025  # 2.5%
        total_requirement = min_capital_ratio + capital_conservation

        result = {
            'rwa_individual': rwa,
            'rwa_total': total_rwa,
            'risk_weights': risk_weights,
            'avg_risk_weight': float(np.mean(risk_weights)),
            'total_exposure': total_exposure,
            'rwa_density': total_rwa / total_exposure if total_exposure > 0 else 0,
            'minimum_capital': total_rwa * min_capital_ratio,
            'total_capital_required': total_rwa * total_requirement,
            'capital_ratio_requirement': total_requirement,
            'approach': approach
        }

        logger.info(f"RWA ({approach}): ${total_rwa:,.0f}, "
                     f"Density: {result['rwa_density']:.2%}")
        return result

    def economic_capital(self, el_individual: np.ndarray,
                          confidence_level: float = 0.999) -> Dict:
        """Compute economic capital using VaR approach."""
        total_el = np.sum(el_individual)

        # Portfolio loss distribution (normal approximation)
        mean_loss = np.mean(el_individual)
        std_loss = np.std(el_individual)

        # VaR at confidence level
        z = stats.norm.ppf(confidence_level)
        var_portfolio = total_el + z * std_loss * np.sqrt(len(el_individual))

        # Expected Shortfall (CVaR)
        es = total_el + std_loss * np.sqrt(len(el_individual)) * (
            stats.norm.pdf(z) / (1 - confidence_level)
        )

        # Economic capital = UL = VaR - EL
        ec = max(var_portfolio - total_el, 0)

        result = {
            'expected_loss': float(total_el),
            'unexpected_loss': float(ec),
            'var': float(var_portfolio),
            'expected_shortfall': float(es),
            'confidence_level': confidence_level,
            'economic_capital': float(ec),
            'ec_as_pct_el': float(ec / total_el * 100) if total_el > 0 else 0
        }

        logger.info(f"Economic Capital @{confidence_level:.1%}: ${ec:,.0f}")
        return result


class StressTester:
    """Stress testing framework for credit risk scenarios."""

    def __init__(self, scenarios: dict = None):
        self.scenarios = scenarios or {
            'baseline': {
                'unemployment_delta': 0.0,
                'gdp_delta': 0.0,
                'property_value_delta': 0.0
            },
            'adverse': {
                'unemployment_delta': 0.03,
                'gdp_delta': -0.02,
                'property_value_delta': -0.20
            },
            'severely_adverse': {
                'unemployment_delta': 0.05,
                'gdp_delta': -0.04,
                'property_value_delta': -0.35
            }
        }
        self.results: Dict = {}

    def stress_pd(self, pd_base: np.ndarray,
                   scenario: dict) -> np.ndarray:
        """Apply stress scenario to PD estimates."""
        # Sensitivity coefficients (calibrated)
        unemployment_coeff = 0.15  # +1pp unemployment -> +15% PD
        gdp_coeff = -0.10          # -1% GDP -> +10% PD
        property_coeff = -0.05     # -1% property -> +5% PD

        stress_multiplier = np.exp(
            unemployment_coeff * scenario['unemployment_delta'] * 100 +
            gdp_coeff * scenario['gdp_delta'] * 100 +
            property_coeff * scenario['property_value_delta'] * 100
        )

        stressed_pd = np.clip(pd_base * stress_multiplier, 0, 1)
        return stressed_pd

    def stress_lgd(self, lgd_base: np.ndarray,
                    scenario: dict) -> np.ndarray:
        """Apply stress to LGD (recovery rates decline in stress)."""
        property_impact = scenario['property_value_delta'] * -0.3
        lgd_stressed = np.clip(lgd_base * (1 + max(property_impact, 0)), 0.01, 0.99)
        return lgd_stressed

    def run_scenarios(self, pd_base: np.ndarray, lgd_base: np.ndarray,
                       ead: np.ndarray) -> Dict:
        """Run all stress scenarios."""
        logger.info("Running stress test scenarios...")
        calculator = BaselCapitalCalculator()

        for name, scenario in self.scenarios.items():
            pd_stressed = self.stress_pd(pd_base, scenario)
            lgd_stressed = self.stress_lgd(lgd_base, scenario)

            el_result = calculator.compute_expected_loss(
                pd_stressed, lgd_stressed, ead
            )
            rwa_result = calculator.compute_rwa(
                pd_stressed, lgd_stressed, ead
            )

            pd_migration = np.mean(pd_stressed) / max(np.mean(pd_base), 1e-6) - 1

            self.results[name] = {
                'scenario_params': scenario,
                'mean_pd': float(np.mean(pd_stressed)),
                'mean_lgd': float(np.mean(lgd_stressed)),
                'total_el': el_result['el_total'],
                'total_rwa': rwa_result['rwa_total'],
                'pd_migration_pct': float(pd_migration * 100),
                'el_increase_pct': float(
                    (el_result['el_total'] / max(
                        self.results.get('baseline', {}).get('total_el', el_result['el_total']),
                        1
                    ) - 1) * 100
                ) if 'baseline' in self.results else 0.0
            }

            logger.info(f"  {name}: Mean PD={np.mean(pd_stressed):.4f}, "
                         f"EL=${el_result['el_total']:,.0f}")

        return self.results

    def get_stress_summary(self) -> pd.DataFrame:
        """Get stress test results summary."""
        rows = []
        for name, result in self.results.items():
            rows.append({
                'scenario': name,
                'mean_pd': result['mean_pd'],
                'mean_lgd': result['mean_lgd'],
                'total_el': result['total_el'],
                'total_rwa': result['total_rwa'],
                'pd_migration_pct': result['pd_migration_pct']
            })
        return pd.DataFrame(rows)
