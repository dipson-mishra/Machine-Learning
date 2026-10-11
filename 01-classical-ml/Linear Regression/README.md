# Linear, Lasso and Ridge Regression for Mexico City Real Estate

Linear Regression tries to fit a line through data by making errors as small as possible. But if features overlap or you have too many variables, the model gets overly sensitive and overfits. Regularization fixes this by adding a penalty for having large coefficients ($\beta$). The hyperparameter $\lambda$ controls how strict that penalty is.

This project explores Mexico City real-estate listings and compares baseline, ordinary least-squares, Ridge, and Lasso regression models. The complete workflow is documented in [`notebooks/real-estate.ipynb`](notebooks/real-estate.ipynb). The Streamlit app and Flask API both use the saved model bundle in `src/real-estate-model.joblib`.

The 3 Models at a Glance
- **Linear Regression**: No penalty. Lets coefficients grow as large as needed to fit the training data.
- **Ridge ($L_2$)**: Penalizes squared values. Shrinks all coefficients close to zero, but keeps every feature. Best for handling correlated variables.
- **Lasso ($L_1$)**: Penalizes absolute values. Forces unhelpful coefficients to exact zero, effectively dropping them. Best for auto-selecting features.
## Repository layout

- `notebooks/real-estate.ipynb` — data exploration, cleaning, visualization, feature engineering, modeling, and evaluation.
- `data/` — five source CSV files (`mexico-city-real-estate-1.csv` through `mexico-city-real-estate-5.csv`).
- `../../requirements.txt` — Python dependencies used by the notebook.

Run the notebook with `notebooks/` as the working directory so its `../data/...` paths resolve correctly. The cleaned combined dataset is written to `data/mexico-real-estate-combined-clean.csv`.

## Workflow

1. Load the five CSV files, validate that their schemas match, and concatenate them.
2. Inspect distributions, missing values, categorical fields, and correlations.
3. Remove irrelevant columns, derive `state`, latitude, and longitude from source location fields, interpolate numeric missing values, and drop rows with unresolved categorical values.
4. Visualize price and surface-area distributions, property/state differences, and outliers. The notebook trims the lowest and highest 10% of price and covered-surface values for the modeling data.
5. Predict `price_aprox_usd` from `surface_covered_in_m2`, `property_type`, and `state`. Categorical features are converted to indicator variables with the most frequent level as the reference category.
6. Drop rows with missing target values, split the modeling dataset into 80% training and 20% test sets (`random_state=42`), and compare models using USD RMSE.

The trainer saves the feature-column order and category levels (including each reference category) alongside the selected model. Prediction uses this saved schema to encode each input. This matters for one-row predictions: fitting `drop_first` encoding on a single row can drop that row's only category and make different choices look identical. Inputs containing category values not present during training are rejected with a clear error instead of silently treated as the reference category. The Streamlit dropdowns are populated from the saved model bundle so they only offer supported values.

## Model results

The models are trained on `log1p(price_aprox_usd)` to reduce target skew. Predictions are converted back with `expm1` before calculating USD RMSE. The training script selects the model using five-fold cross-validated USD RMSE on the training split, then reports test RMSE on the held-out test split. The notebook compares a log-scale baseline with ordinary least squares, RidgeCV, and LassoCV models; rerun the notebook to regenerate the current metrics after preprocessing or dependency changes.

Model selection is based on five-fold cross-validation RMSE on the training split, with the test set reserved for final reporting. The current test R² is about 0.22, so the model captures some price variation but its estimates remain limited. Remaining error reflects information not included in the three-feature model, such as finer-grained location, amenities, and property age.

# Model Comparison

| Model | CV RMSE | Train RMSE | Test RMSE | Test R² |
| :--- | ---: | ---: | ---: | ---: |
| **Linear Regression** | 101,474.12 | 101,292.42 | 101,576.56 | 0.2197 |
| **Ridge (CV-tuned, selected)** | 101,466.34 | 101,287.29 | 101,576.36 | 0.2197 |
| **Lasso (CV-tuned)** | 101,480.00 | 101,291.61 | 101,578.50 | 0.2197 |

Metrics are in USD. The log-price baseline shown in earlier notebook results is not part of the three-model comparison produced by `src/train.py`.

## Tools

Python, Flask, pandas, NumPy, matplotlib, seaborn, scikit-learn, category-encoders, and Jupyter Notebook.

## Flask API

Install the project dependencies and start the API from this project directory (`01-classical-ml/Linear Regression`):

```bash
python -m pip install -r requirements.txt
python -m src.app
```

The service listens on port 5000. `GET /health` reports whether the service is running. `POST /predict` accepts one property:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"surface_covered_in_m2": 80, "property_type": "apartment", "state": "Distrito Federal"}'
```

For batch predictions, send `{ "properties": [ ... ] }` with the same three fields on each property. The response contains `predicted_price_aprox_usd` values. The API loads `src/real-estate-model.joblib`; run `python src/train.py` to create or refresh the model bundle.

## Streamlit website

From this project directory, install the dependencies and start the website with:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The property type and state choices are read from the trained model bundle. Changing a supported category changes its encoded model input and can change the estimate; the size of the effect depends on the learned coefficient. The trained category levels come from the filtered training data, so the app may offer fewer regions than the full dataset.
