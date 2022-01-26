import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd

file_stem = 'crabs_pca'
feature_cols = ['pc_1', 'pc_2']
klass_cols = ['sex', 'sp']

print(f"K-Means usando el dataset {file_stem}.")

# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")

# Separate feature columns.
X = data_frame.loc[:, feature_cols]

# Apply K-Means.
prediction = KMeans(n_clusters=2).fit_predict(X)

# Add prediction to data frame.
data_frame['prediction'] = prediction

# Count classes per cluster.
hit_list = []
for klass in klass_cols:
    # TODO: Seguro se mejora usando dataframe.
    hit_count = 0
    pred_to_klass = data_frame[klass].unique()

    for pred, klass in zip(prediction, data_frame[klass]):
        if(klass == pred_to_klass[pred]):
            hit_count += 1
    
    # The clustering has no way to distinguish between classes, so 0 could mean any
    # class. 9 hits and 1 miss and 9 misses and 1 hit are the same result.
    hit_list.append(max(hit_count, len(data_frame) - hit_count))

# Graph prediction.
color_list = list(mcolors.TABLEAU_COLORS.keys())
for pred in data_frame['prediction'].unique():
    idx_list = data_frame['prediction'] == pred
    plt.scatter(
        data_frame.loc[idx_list, 'pc_1'],
        data_frame.loc[idx_list, 'pc_2'],
        marker="o",
        color= color_list[pred]
    )
plt.title(f"K-Means for dataset {file_stem}")
plt.savefig(fname=f"{file_stem}_k_means")


for klass, hits in zip(klass_cols, hit_list):
    print(f"{klass} hits: {hits}")

