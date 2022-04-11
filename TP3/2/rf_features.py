from black import out
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

file_stem = "RRL"
min_depth = 1
n_estimators = 1000
n_iterations = 5

print(f"Random Forest usando el dataset {file_stem}.")

# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")
data_frame = data_frame.iloc[:, 1:]  # Remove first column.

features = data_frame.iloc[:, :-1]
target = data_frame.iloc[:, -1]

oob_error = []
n_features = len(features.columns)
n_features_list = []
while n_features:
    error = 0

    for _ in range(n_iterations):
        classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_features=n_features, oob_score=True
        )
        # Fit the model with the data.
        classifier.fit(features, target)
        # Record the oob error.
        error += 1 - classifier.oob_score_
        print(f"{n_features} {_}")

    oob_error.append(error / n_iterations)
    n_features_list.append(n_features)
    n_features //= 2
oob_error.reverse()
n_features_list.reverse()

# Write output file.
out_file = open(f"{file_stem}_oob_err.txt", "w")
for n_features, error in zip(n_features_list, oob_error):
    out_file.write(f"{n_features}, {error}\n")
out_file.close()

# Graph prediction errors.
color_list = list(mcolors.TABLEAU_COLORS.keys())
x_label = "max features per split"
y_label = "oob error"
plt.plot(
    n_features_list,
    oob_error,
    marker="o",
    color=color_list[0],
)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.xticks(n_features_list)
plt.title(f"RandomForestClassifier using dataset {file_stem}")
plt.savefig(fname=f"{file_stem}_rf")
