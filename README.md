### Sonar Mines vs Rocks Classification – scikit-learn

This project trains and evaluates a classifier to distinguish “Mine” (M) vs “Rock” (R) objects from Sonar measurements. The learning pipeline is StandardScaler → LDA → Logistic Regression, with hyperparameter search via cross-validation (GridSearchCV).

### Repository contents

- `Sonar_data.csv`: dataset (60 feature columns + 1 target column M/R)
- `sonar_dataset_overview.py`: quick data overview (shape, stats, class distribution, per-class means)
- `main.py`: training with a pipeline and GridSearchCV, then classification report and confusion matrix

### Requirements

- Python 3.9+ recommended
- Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`

Quick setup in a virtual environment (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy scikit-learn
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

- Best hyperparameters found by GridSearchCV
- Best mean cross-validation accuracy
- Classification report on the test set
- Confusion matrix

### Model notes

- Pipeline: `StandardScaler` → `LDA(n_components=1)` → `LogisticRegression`.
- Train/test split: 80%/20% (`random_state=0`).
- Hyperparameter search: `GridSearchCV(cv=10, scoring='accuracy', n_jobs=-1)`. Depending on your scikit‑learn version, some `solver`/`penalty`/`dual` combinations are unsupported. If you see errors like “combination is not supported”, restrict the grid (e.g., remove `elasticnet` or use `solver='saga'` only when using `elasticnet`, and consider specifying `l1_ratio`).

### Dataset structure

- 60 numeric feature columns (indices 0 to 59)
- 1 target column (index 60) with labels: `M` (Mine) or `R` (Rock)

### Possible improvements

- Explore compatible solver/penalty combos (e.g., `saga` for `elasticnet`)
- Try other models (SVM, RandomForest, Gradient Boosting)
- Stratified cross-validation and Bayesian optimization for hyperparameters
- ROC/PR curves and additional metrics (AUC, per-class F1)

### Credits

- Data inspired by the UCI ML Repository Sonar (Mines vs Rocks) dataset.






