# Machine Learning Portfolio & Experiments

Welcome to my Machine Learning repository! This workspace serves as a hands-on portfolio and lab for various ML projects, ranging from foundational classical machine learning algorithms to advanced deep learning architectures and deployment strategies.

> **Note:** This repository is an active work in progress. Projects will range from quick technical experiments and model implementations to fully polished end-to-end applications.

## Live demo

Try the [Mexico City Real Estate Price Predictor](https://l-regression.streamlit.app), powered by the Ridge regression model in this repository.

## Setup

The repository uses one shared dependency file. Create and activate a virtual environment from the repository root, then install all project dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Individual projects may need only a subset of these packages, but use this root `requirements.txt` for local development and Streamlit Community Cloud deployments.

---

## 🎯 Objectives & Vision

- **Broad Coverage:** Implement solutions using a wide variety of techniques—starting with classical ML and expanding into deep learning, NLP, computer vision, and reinforcement learning.
- **Hands-on Experimentation:** Focus on practical model building, feature engineering, and hyperparameter tuning.
- **Progress Tracking:** Document key learnings, performance metrics, and iteration steps across different domain problems.

---

## 🛠️ Tech Stack & Tools

- **Languages:** Python
- **Core Libraries:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Frameworks (Upcoming):** PyTorch / TensorFlow

---

## 📁 Repository Structure

Each project lives in its own directory containing dataset references, notebooks, source code, and project-specific documentation:

```text
.
├── 01-classical-ml/          # Regression, Classification, Clustering, Ensembles
│   └── project-name/
│       ├── data/             # Datasets or setup scripts
│       ├── notebooks/        # Exploratory Data Analysis & experiments
│       ├── src/              # Clean, modular Python scripts
│       └── README.md         # Project overview & results
│
├── 02-deep-learning/         # Neural Networks, Computer Vision, NLP (Planned)
│   └── project-name/
│
├── requirements.txt          # Shared dependencies for all projects
└── README.md                 # Main repository overview
