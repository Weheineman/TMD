import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

file_stem = "diag"
feature_cols = ["V1", "V2"]
target_col = "class"
min_depth = 1
max_depth = 20
n_estimators = 200

print(f"Boosting usando el dataset {file_stem}.")

# Read data.
train_df = pd.read_csv(f"{file_stem}_train.csv")
test_df = pd.read_csv(f"{file_stem}_test.csv")

depth_error = []
for depth in range(min_depth, max_depth + 1):
    classifier = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=n_estimators,
    )
    # Fit the model with the training data.
    classifier.fit(train_df.loc[:, feature_cols], train_df.loc[:, target_col])
    # Record the test error.
    depth_error.append(
        1 - classifier.score(test_df.loc[:, feature_cols], test_df.loc[:, target_col])
    )

# Graph prediction errors.
color_list = list(mcolors.TABLEAU_COLORS.keys())
x_label = "decision tree depth"
y_label = "test error"
plt.plot(
    range(min_depth, max_depth + 1),
    depth_error,
    marker="o",
    color=color_list[0],
)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.xticks(range(min_depth, max_depth + 1))
plt.title(f"AdaBoostClassifier using dataset {file_stem}")
plt.savefig(fname=f"{file_stem}_boosting")
