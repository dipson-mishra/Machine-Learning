# Linear, Lasso and Ridge Regression for Mexico City Real Estate

This project explores Mexico City real-estate listings and compares baseline, ordinary least-squares, Ridge, and Lasso regression models. The complete workflow is documented in [`notebooks/real-estate.ipynb`](notebooks/real-estate.ipynb).

## Repository layout

- `notebooks/real-estate.ipynb` — data exploration, cleaning, visualization, feature engineering, modeling, and evaluation.
- `data/` — five source CSV files (`mexico-city-real-estate-1.csv` through `mexico-city-real-estate-5.csv`).
- `requirements.txt` — Python dependencies used by the notebook.

Run the notebook with `notebooks/` as the working directory so its `../data/...` paths resolve correctly. The cleaned combined dataset is written to `data/mexico-real-estate-combined-clean.csv`.

## Workflow

1. Load the five CSV files, validate that their schemas match, and concatenate them.
2. Inspect distributions, missing values, categorical fields, and correlations.
3. Remove irrelevant columns, derive `state`, latitude, and longitude from source location fields, interpolate numeric missing values, and drop rows with unresolved categorical values.
4. Visualize price and surface-area distributions, property/state differences, and outliers. The notebook trims the lowest and highest 10% of price and covered-surface values for the modeling data.
5. Predict `price_aprox_usd` from `surface_covered_in_m2`, `property_type`, and `state`. Categorical features are converted to indicator variables with the most frequent level as the reference category.
6. Drop rows with missing target values, split the modeling dataset into 80% training and 20% test sets (`random_state=42`), and compare models using USD RMSE.

## Model results

The models are trained on `log1p(price_aprox_usd)` to reduce target skew. Predictions are converted back with `expm1` before calculating USD RMSE. The notebook compares a log-scale baseline with ordinary least squares, RidgeCV, and LassoCV models; rerun the notebook to regenerate the current metrics after preprocessing or dependency changes.

Model selection should be based on validation or cross-validation results, with the test set reserved for the final report. Remaining error reflects information not included in the three-feature model, such as finer-grained location, amenities, and property age.

# Model Comparison

| Model | Train RMSE | Test RMSE |
| :--- | :---: | :---: |
| **Baseline** | 115,906.39 | 111,243.98 |
| **Linear Regression** | 95,877.50 | 93,179.66 |
| **Ridge (CV-tuned)** | 95,877.63 | 93,172.18 |
| **Lasso (CV-tuned)** | 95,877.50 | 93,179.66 |

## Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, category-encoders, and Jupyter Notebook.
