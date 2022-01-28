import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
import pandas as pd

file_stem = "crabs_log_pca_scale"
feature_cols = [f"pc_{idx}" for idx in range(1, 6)]
klass_cols = ["sex", "sp"]
method = KMeans
method_name = "k_means"
n_clusters = 2

# https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cont_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cont_matrix, axis=0)) / np.sum(cont_matrix)


print(f"{method_name} usando el dataset {file_stem}.")

# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")

# Separate feature columns.
X = data_frame.loc[:, feature_cols]

# Apply clustering.
prediction = method(n_clusters=n_clusters).fit_predict(X)

# Add prediction to data frame.
data_frame["prediction"] = prediction

# Score the clustering for each target feature.
purity_score_list = [
    purity_score(data_frame[klass], prediction) for klass in klass_cols
]

# Graph clustering prediction.
color_list = list(mcolors.TABLEAU_COLORS.keys())
x_label = "pc_1"
y_label = "pc_2"
for pred in data_frame["prediction"].unique():
    idx_list = data_frame["prediction"] == pred
    plt.scatter(
        data_frame.loc[idx_list, x_label],
        data_frame.loc[idx_list, y_label],
        marker="o",
        color=color_list[pred],
    )
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f"{method_name} for dataset {file_stem}")
plt.savefig(fname=f"{file_stem}_{method_name}")


for klass, purity in zip(klass_cols, purity_score_list):
    print(f"{klass} score: {purity}")
