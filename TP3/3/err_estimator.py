import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

file_stem = "lampone"
estimator = SVC(kernel="rbf")
n_splits = 5
param_grid = {
    "gamma": np.logspace(-10, 10, 21),
    "C": np.logspace(-10, 10, 21),
}

print(f"SVM RBF usando el dataset {file_stem}.")

# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")
data_frame = data_frame.iloc[:, 1:]  # Remove first column.

features = data_frame.loc[:, [col for col in data_frame if col.startswith("m")]]
target = data_frame["N_tipo"]

estimated_err = 0
k_fold = StratifiedKFold(n_splits, shuffle=True)
for train_index, val_index in k_fold.split(features, target):
    random_forest = GridSearchCV(estimator, param_grid)
    random_forest.fit(features.iloc[train_index], target.iloc[train_index])
    val_score = random_forest.score(features.iloc[val_index], target.iloc[val_index])
    estimated_err += 1 - val_score
    print(
        f"Validation error: {1 - val_score} with params: {random_forest.best_params_}"
    )
estimated_err /= k_fold.get_n_splits()

print(f"Estimated test error: {estimated_err}")
