# Architecture Reference

## System Architecture

```
+------------------------------------------------------------+
|               Credit Risk Platform                         |
|                main.py (Pipeline)                          |
+-----------------------------+------------------------------+
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
+-------------------+ +----------------+ +------------------+
|  src/data/        | | src/features/  | |  src/models/     |
|  data_generator   | | feature_eng    | |  pd_model        |
|                   | |                | |  lgd_model       |
|  - 500K+ loans    | |  - 100+ feats  | |  ead_model       |
|  - Demographics   | |  - WoE / IV    | |                  |
|  - Credit history | |  - KS validate | |  - Logistic Reg  |
|  - Macro factors  | |  - Interactions| |  - XGBoost       |
|  - Default events | |  - Polynomials | |  - Beta Reg      |
|  - LGD / EAD      | |  - Binning     | |  - Cox PH        |
+-------------------+ +----------------+ |  - Vasicek Model |
                                          +--------+---------+
                                                   |
                            +----------------------+
                            |                      |
                            v                      v
              +----------------------+  +-----------------------+
              | src/validation/      |  | src/regulatory/       |
              | model_validation     |  | capital_calculation   |
              |                      |  |                       |
              | - Hosmer-Lemeshow    |  | - EL = PD x LGD x EAD|
              | - PSI monitoring     |  | - IRB Advanced RWA    |
              | - Binomial backtest  |  | - Economic Capital    |
              | - Traffic light      |  | - VaR / CVaR          |
              | - Migration matrix   |  | - Stress testing      |
              | - ROC / CAP curves   |  | - Basel III compliance|
              +----------------------+  +-----------------------+
```

## Data Flow

1. **Data Generation**: Synthetic borrower profiles with realistic distributions
   - Credit scores (300 to 850, log-normal)
   - Loan amounts ($1K to $500K, log-normal)
   - Default outcomes calibrated via logistic model

2. **Feature Engineering**: Raw to 100+ model-ready features
   - Financial ratios (DTI, LTV, utilization)
   - Interaction terms (score x utilization)
   - Polynomial / log / sqrt transforms
   - WoE encoding with IV ranking

3. **Model Training**: Three parallel model streams
   - PD: XGBoost vs Logistic, auto-select by AUC
   - LGD: Beta regression + Gradient Boosting ensemble
   - EAD: Cox survival + CCF for revolving facilities

4. **Validation**: Regulatory-grade model testing
   - Discrimination: AUC, Gini, KS, Accuracy Ratio
   - Calibration: Hosmer-Lemeshow, PSI, binomial test
   - Stability: 24-month PSI monitoring, migration matrices

5. **Capital Calculation**: Basel III outputs
   - Expected Loss with confidence intervals
   - IRB risk weights with maturity adjustment
   - Stress scenarios: baseline, adverse, severely adverse

## Key Algorithms

| Component | Algorithm | Reference |
|-----------|-----------|-----------|
| PD | XGBoost Classifier | Chen & Guestrin (2016) |
| PD | Logistic Regression | Cox (1958) |
| PD Calibration | Vasicek Single-Factor | Vasicek (2002) |
| LGD | Beta Regression | Ferrari & Cribari-Neto (2004) |
| EAD | Cox Proportional Hazards | Cox (1972) |
| Validation | Hosmer-Lemeshow | Hosmer & Lemeshow (1980) |
| Capital | Basel III IRB Formula | BCBS 128 (2006) |
| Stress | CCAR/DFAST Framework | Federal Reserve (2020) |
