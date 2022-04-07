from re import S
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import fowlkes_mallows_score
import pandas as pd

file_stem = "lampone_log_scale_pca"
feature_prefix = "pc_"
method = KMeans
method_name = "k_means"
max_clusters = 10
iterations = 20

# Calculates the similarity score for n_clusters clusters using iterations subsamples.
def similarity_score(data_frame, n_clusters):
    sample_list = []
    for iter in range(iterations):
        # Use a 90% sample size.
        df_sample = data_frame.sample(frac=0.9)
        # The cluster column names are all different so that the dataframes can be
        # merged with an inner join later.
        df_sample[f"cluster{iter}"] = method(n_clusters=n_clusters).fit_predict(
            df_sample
        )
        sample_list.append(df_sample)

    # Make a list with the stability score for every pair of samples.
    score_list = []
    for idx_1 in range(iterations):
        for idx_2 in range(idx_1 + 1, iterations):
            df_merge = pd.merge(
                sample_list[idx_1], sample_list[idx_2], on=list(data_frame.columns)
            )
            score_list.append(
                fowlkes_mallows_score(
                    df_merge[f"cluster{idx_1}"], df_merge[f"cluster{idx_2}"]
                )
            )

    return score_list


# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")
feature_cols = [col for col in data_frame if col.startswith(feature_prefix)]

print(f"Stability con {method_name} usando el dataset {file_stem}.")

# Separate feature columns.
features = data_frame.loc[:, feature_cols]

# Calculate similarity scores for each cluster amount.
similarity = [
    similarity_score(features, n_clusters) for n_clusters in range(2, max_clusters + 1)
]

# Graph cumulative gap statistic histogram.
bins = np.arange(0.5, 1.01, 0.01)
color_list = list(mcolors.TABLEAU_COLORS.keys())
for n_clusters in range(2, max_clusters + 1):
    plt.hist(
        similarity[n_clusters - 2],
        bins=bins,
        density=True,
        cumulative=True,
        histtype="step",
        label=f"{n_clusters}",
        color=color_list[n_clusters - 2],
    )
plt.legend()
plt.xlabel("Fowlkes Mallows similarity")
plt.ylabel("cumulative")
plt.title(f"Cumulative FM similarity using {method_name} for dataset {file_stem}")
plt.savefig(fname=f"{file_stem}_stability_{method_name}")
