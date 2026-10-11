# K-Nearest Neighbors Classifier

This project trains a KNN classifier to distinguish gamma-ray events (`g`) from
hadron events (`h`) in the [UCI MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).
The target is encoded as `1` for gamma and `0` for hadron. The notebook in
`notebooks/notebook.ipynb` contains the exploratory analysis and original model
workflow; the scripts in `src/` provide a repeatable command-line workflow.

## Project structure

```text
data/                 Train, validation, and test CSV files
models/               Trained model and evaluation metrics (created on training)
notebooks/            Exploratory notebook
src/prepare_data.py   Download data and create stratified splits
src/train.py          Fit and evaluate the KNN classifier
../../requirements.txt Shared repository dependencies
```

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment from the
repository root, then install the shared dependencies:

```bash
python -m venv .venv
# Windows PowerShell (from repository root)
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the KNN commands below from this project directory. From the repository root, switch into it with `cd "01-classical-ml/KNN"` (or the equivalent for your shell).

## Prepare the data

Download the dataset from the UCI repository and create stratified splits:

```bash
python src/prepare_data.py
```

This writes `data/train.csv`, `data/valid.csv`, and `data/test.csv`. Each file
contains the ten telescope measurements and a `class` column. The split sizes
are 60% training, 20% validation, and 20% test, with a fixed random seed (`42`)
for reproducibility. Choose another seed with
`python src/prepare_data.py --random-state 7`. The preparation step requires
internet access to fetch the
dataset. It replaces the existing split CSVs.

## Train and evaluate

```bash
python src/train.py
```

The script fits a standard scaler on training features, oversamples the
training set to address class imbalance, and fits a 10-neighbor classifier.
It reports classification metrics on the validation and test splits and saves
the fitted pipeline to `models/knn.joblib` and the reports to
`models/metrics.json`.

To choose another neighbor count:

```bash
python src/train.py --neighbors 7
```

Set the oversampling seed with `python src/train.py --random-state 7`. KNN
prediction itself is deterministic and does not provide a `random_state`
parameter.

The saved pipeline includes scaling, so it accepts raw feature columns in the
same order as the CSV files when used for predictions. Keep the test split for
final evaluation; use the validation results when comparing model settings.

## Results

With 10 neighbors and random seed `42`, the run produced:

| Split | Accuracy | Hadron F1 | Gamma F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 81.55% | 0.7425 | 0.8562 | 0.7993 |
| Test | 81.73% | 0.7444 | 0.8578 | 0.8011 |

The full precision, recall, and support metrics are saved in
`models/metrics.json` after training.
