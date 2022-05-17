import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

file_stem = "RRL"
estimator = SVC(kernel="poly")
n_splits = 5
param_grid = {
    "degree": range(1, 4),
    "C": [0.5, 1.5, 2.5],
}

print(f"SVM Polinomial usando el dataset {file_stem}.")

# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")
data_frame = data_frame.iloc[:, 1:]  # Remove first column.

features = data_frame.iloc[:, :-1]
target = data_frame["Tipo"]

estimated_err = 0
k_fold = StratifiedKFold(n_splits, shuffle=True)
for train_index, val_index in k_fold.split(features, target):
    model = GridSearchCV(estimator, param_grid)
    model.fit(features.iloc[train_index], target.iloc[train_index])
    val_score = model.score(features.iloc[val_index], target.iloc[val_index])
    estimated_err += 1 - val_score
    print(
        f"Validation error: {1 - val_score} with params: {model.best_params_}"
    )
estimated_err /= k_fold.get_n_splits()

print(f"Estimated test error: {estimated_err}")
