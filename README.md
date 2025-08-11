### Sonar Mines vs Rocks Classification – scikit-learn

This project trains and evaluates a machine learning pipeline to distinguish “Mine” (M) vs “Rock” (R) objects from Sonar measurements. The pipeline uses StandardScaler → LDA → a tunable classifier. Models explored via GridSearchCV include Logistic Regression, SVM (linear and RBF), RandomForest, and XGBoost.

### Repository contents

- `Sonar_data.csv`: dataset (60 feature columns + 1 target column M/R)
- `sonar_dataset_overview.py`: quick data overview (shape, stats, class distribution, per-class means)
- `main.py`: training with a pipeline and GridSearchCV, model selection, and evaluation
- `best_model.pkl`: serialized best pipeline saved after training

### Requirements

- Python 3.9+ recommended
- Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `joblib`

Quick setup in a virtual environment (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost joblib
```

### Usage

1) Dataset overview

```powershell
python sonar_dataset_overview.py
```

Shows: dataset shape, statistical summary, M/R distribution, and per-class feature means.

2) Train + evaluate the model

```powershell
python main.py
```

Outputs:

- Best model (full pipeline)
- Best hyperparameters found by GridSearchCV
- Best mean cross-validation accuracy
- Classification report on the test set
- Confusion matrix
- Saves the best pipeline to `best_model.pkl`

### Model notes

- Pipeline: `StandardScaler` → `LDA(n_components=1)` → `classifier` (one of LogisticRegression, SVC linear/RBF, RandomForest, XGBoost)
- Train/test split: 80%/20% (`random_state=0`).
- Hyperparameter search: `GridSearchCV(cv=10, scoring='accuracy', n_jobs=-1)` over multiple model families via the `classifier` step.

### Dataset structure

- 60 numeric feature columns (indices 0 to 59)
- 1 target column (index 60) with labels: `M` (Mine) or `R` (Rock)

### Possible improvements

- Calibrate probabilities (Platt scaling/Isotonic) for SVM and XGBoost
- Add feature importance and permutation importance analyses
- Stratified cross-validation and Bayesian or randomized hyperparameter search
- ROC/PR curves and additional metrics (AUC, per-class F1)

### Credits

- Data inspired by the UCI ML Repository Sonar (Mines vs Rocks) dataset.






