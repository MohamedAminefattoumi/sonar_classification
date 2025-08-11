#importing the libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Importing the dataset 
dataset = pd.read_csv('Sonar_data.csv', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Encoding the target column 
en = LabelEncoder()
y_train = en.fit_transform(y_train)
y_test = en.transform(y_test)

# Creating the pipeline with scaling, LDA and Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA(n_components=1)),
    ('logreg', LogisticRegression(random_state=0))
])

# Parameters grid to tune Logistic Regression hyperparameters
params = {
    'logreg__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'logreg__solver': ['liblinear', 'lbfgs'],
    'logreg__penalty': ['l1','l2','elasticnet'],
    'logreg__dual': [True , False],

}

# GridSearchCV with 5-fold CV to find best hyperparameters
grid = GridSearchCV(pipeline, param_grid=params, cv=10, scoring='accuracy', n_jobs=-1)

# Training with GridSearchCV
grid.fit(X_train, y_train)

# Best parameters and best cross-validation accuracy
print("Best parameters found:", grid.best_params_)
print(f"Best cross-validation accuracy: {grid.best_score_ * 100:.2f} %")

# Predicting on test set using the best model
y_pred = grid.predict(X_test)

# Evaluation metrics on test set
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
