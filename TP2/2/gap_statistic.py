import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

file_stem = "iris_log_scale_pca"
feature_prefix = "pc_"
method = KMeans
method_name = "k_means"
max_clusters = 10
uniform_count = 20


# Generates a uniform distribution with the same amount of
# data points in the hyperrectangle of the source data frame.
def generate_uniform(source_data: npt.ArrayLike) -> npt.ArrayLike:
    uniform_data = source_data.copy().transpose()
    # Iterate over the columns (features) of the original matrix.
    for idx, row in enumerate(uniform_data):
        uniform_data[idx] = np.random.uniform(row.min(), row.max(), len(row))
    return uniform_data.transpose()


# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")
feature_cols = [col for col in data_frame if col.startswith(feature_prefix)]

print(f"Gap statistic con {method_name} usando el dataset {file_stem}.")

# Separate feature columns.
features = data_frame.loc[:, feature_cols]

# Calculate log inertia for different cluster amounts.
data_inertia = [
    np.log(method(n_clusters=n_clusters).fit(features).inertia_)
    for n_clusters in range(1, max_clusters + 1)
]

# Apply PCA.
features = PCA().fit_transform(features)

# Calculate log inertia for uniform distributions.
uniform_inertia = np.zeros((max_clusters, uniform_count))
for cluster_idx in range(max_clusters):
    for iteration in range(uniform_count):
        uniform_data_frame = generate_uniform(features)
        uniform_inertia[cluster_idx][iteration] = np.log(
            method(n_clusters=cluster_idx + 1).fit(uniform_data_frame).inertia_
        )

# Calculate mean and gap statistic.
uniform_inertia_mean = np.mean(uniform_inertia, axis=1)
gap = uniform_inertia_mean - data_inertia

# Calculate standard deviation and adjusted standard deviation.
uniform_inertia_sd = np.zeros(max_clusters)
for cluster_idx in range(max_clusters):
    uniform_inertia_sd[cluster_idx] = np.sqrt(
        np.mean(
            np.square(uniform_inertia[cluster_idx] - uniform_inertia_mean[cluster_idx])
        )
    )
uniform_inertia_s = uniform_inertia_sd * np.sqrt(1 + 1 / uniform_count)

# Graph gap statistic.
plt.errorbar(np.arange(max_clusters) + 1, gap, uniform_inertia_s, color="k")
plt.xlabel("number of clusters")
plt.ylabel("gap")
plt.title(f"Gap statistic using {method_name} for dataset {file_stem}")
plt.savefig(fname=f"{file_stem}_gap_{method_name}")

# Use gap statistic to select cluster size.
cluster_size = max_clusters
for k in range(1, max_clusters):
    if gap[k - 1] >= gap[k] - uniform_inertia_s[k]:
        # This is because gap[k] is the gap statistic for k+1 clusters.
        cluster_size = k
        break
print(f"cluster size: {cluster_size}")
