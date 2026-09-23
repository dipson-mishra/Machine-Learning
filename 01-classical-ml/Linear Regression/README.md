# Mexico City Real Estate Price Prediction — Linear Regression

A complete data science pipeline built on public Mexico City real estate listings, focused on demonstrating a rigorous linear regression workflow: cleaning → EDA → feature engineering → modeling → evaluation.

---

## Project Scope

This project is intentionally scoped to **linear regression** as the modeling technique. The goal isn't to chase the best possible predictive score — it's to show the full pipeline done correctly: careful preprocessing, justified feature choices, proper train/test evaluation, and honest interpretation of results (including the model's limitations).

---

## The Data

Five CSV files covering the Mexico City metro area, combined into a single dataset of **23,140 raw listings** across **16 features**.

---

## Pipeline

### 1. Data Cleaning & Preprocessing

- Loaded all five files, verified schemas matched, and concatenated into one dataframe.
- Dropped redundant/irrelevant columns (`operation`, `currency`, `price`, `floor`, `expenses`, `properati_url`, etc.).
- Engineered `state`, `lat`, and `lon` from the combined location fields (`place_with_parent_names`, `lat-lon`).
- Handled missing values: linear interpolation for numeric features (`surface_covered_in_m2`, `price_aprox_usd`), row-drop for unresolvable categorical nulls.
- Exported the cleaned dataset as `mexico-real-estate-combined-clean.csv`.

### 2. Visualization & Outlier Treatment

- Histograms of price and surface area (log-transformed via `np.log1p` to address right skew).
- Boxplots of price by property type and by state to inspect spread and outliers.
- Removed extreme values by trimming the top/bottom 10th percentile on price and surface area.
- Scatter plot + regression line (`seaborn.regplot`) to visualize the surface-area/price relationship.

### 3. Correlation & Variable Analysis

- Correlation matrix and heatmap across numeric features.
- Correlation between surface area and price, computed overall and grouped by state and property type.
- Investigated price-per-m² vs. surface area to check whether larger properties are cheaper per unit (they are — moderate negative correlation).

### 4. Predictive Modeling — Linear Regression

- **Features**: `surface_covered_in_m2` (continuous), `property_type` and `state` (one-hot encoded, reference-level baseline set to the most frequent category to avoid the dummy trap and reduce multicollinearity).
- **Target**: `price_aprox_usd`, log-transformed (`log1p`) to stabilize variance given its right skew.
- **Model**: `sklearn.linear_model.LinearRegression`, fit on an 80/20 train/test split (`random_state=42`).
- **Evaluation**: MAE, RMSE, RMSLE, and R² — computed on both the log scale (what the model optimizes) and back-transformed USD scale (for interpretability).
- Coefficients converted to percentage price effects (`exp(coef) - 1`) for interpretation — e.g., how much each additional m² or each property type/state shifts predicted price.

---

## Results

| Metric | Train | Test |
|---|---|---|
| MAE (USD) | $73,245 | $70,769 |
| RMSE (USD) | $101,787 | $98,888 |
| RMSLE | 0.57 | 0.56 |
| R² (log scale) | 0.30 | 0.29 |

- Train and test performance are close, which rules out overfitting — the model is **underfitting**: it's missing predictive signal, not memorizing noise.
- Relative MAE (~54% of median price) confirms the model captures a real but weak relationship between surface area, property type, state, and price.

---

## What This Demonstrates

- End-to-end ownership of a regression pipeline: cleaning decisions, encoding choices, and evaluation are all deliberate and explained, not defaults.
- Correct handling of skewed targets (log-transform + back-transform for interpretable error metrics).
- Understanding of *why* the model underperforms (omitted variables — location granularity beyond state-level, property age, amenities) rather than just reporting a score.
- Comfort reading and communicating regression diagnostics: coefficient interpretation, residual behavior, train/test comparison.

---

## Tools Used

Python, pandas, numpy, matplotlib, seaborn, scikit-learn, Jupyter Notebook