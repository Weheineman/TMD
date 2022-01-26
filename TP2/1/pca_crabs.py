import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file_stem = '1/crabs'
feature_cols = ['FL', 'RW', 'CL', 'CW', 'BD']
klass_cols = ['sex', 'sp']

print(f"PCA usando el dataset {file_stem}.")

# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")

# Separate feature columns.
X = data_frame.loc[:, feature_cols]

# Apply log to the features (because the statement recommends it).
X = np.log(X)

# Normalize features.
X = StandardScaler().fit_transform(X)

# Apply PCA.
principal_components = PCA(n_components=2).fit_transform(X)

# Build PC dataframe.
pc_data_frame = pd.DataFrame(data = principal_components, columns = ['pc_1', 'pc_2'])
pc_data_frame = pd.concat([pc_data_frame, data_frame.loc[:, klass_cols]], axis=1)

# Graph PCA.
for klass_name in klass_cols:
    klass_list = data_frame[klass_name].unique()
    color_list = list(mcolors.TABLEAU_COLORS.keys())
    for klass, color in zip(klass_list, color_list):
        idx_list = pc_data_frame[klass_name] == klass
        plt.scatter(
            pc_data_frame.loc[idx_list, 'pc_1'],
            pc_data_frame.loc[idx_list, 'pc_2'],
            marker="o",
            color= color
        )
    plt.title(f"PCA for dataset {file_stem} with target {klass_name}")
    plt.legend(klass_list)
    plt.savefig(fname=f"{file_stem}_{klass_name}_pca_graph")

# Export data.
pc_data_frame.to_csv(f"{file_stem}_pca.csv")