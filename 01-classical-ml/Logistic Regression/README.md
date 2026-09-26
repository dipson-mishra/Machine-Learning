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
   * If the probability is greater than or equal to the threshold, the prediction is classified as the positive class.

---



