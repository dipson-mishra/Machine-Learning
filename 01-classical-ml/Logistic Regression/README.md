# Logistic Regression 

**Logistic Regression** is a regression model that predicts the probability of a categorical outcome (most commonly binary, such as positive/negative).

---

## 🔑 Key Characteristics

* **Categorical Label:** Typically predicts binary outcomes (e.g., spam vs. non-spam). *Multinomial logistic regression* extends this to more than two categories.
* **Loss Function:** Uses **Log Loss** (Binary Cross-Entropy) during training.
* **Architecture:** Linear model structure that passes raw predictions ($y'$) into a **Sigmoid activation function** to map output values between 0 and 1.

---

## ⚙️ Architecture \& Prediction Flow

```
Linear Combination (y') ➔ Sigmoid Function ➔ Probability (0 to 1) ➔ Classification Threshold
```

1. **Raw Prediction ($y'$):** Calculated using a linear combination of input features 
2. **Sigmoid Transformation:** Transforms $y'$ into a valid probability:
3. **Classification Decision:** The output probability is compared against a defined **classification threshold** (e.g., 0.50):

---

## Sentiment analysis example

The `notebooks/sentiment_analysis.ipynb` notebook uses TF–IDF features and logistic regression to classify text as negative (`-1`), neutral (`0`), or positive (`1`). It checks the input data, removes rows missing text or labels, normalizes text, and makes a stratified train/test split. The TF–IDF vocabulary is fitted on the training data only. Evaluation includes accuracy, per-class precision, recall, F1, and a confusion matrix.

### Input data

Place the dataset at `data/sentiment.csv`. It must contain `clean_text` and `category` columns, with sentiment labels `-1`, `0`, and `1`. The dataset is excluded from Git because it is local input data; use data you have permission to use and share.

### Run the notebook

From the repository root, install the dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook notebooks/sentiment_analysis.ipynb
```

The notebook uses a fixed random seed and a stratified 80/20 split. Scores describe this dataset and split and are a baseline, not a guarantee of performance on new text.



