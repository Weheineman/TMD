import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file_stem = "crabs"
# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")

feature_cols = ["FL", "RW", "CL", "CW", "BD"]
klass_cols = ["sex", "sp"]

# Separate feature columns.
X = data_frame.loc[:, feature_cols]

# Processed file stem.
file_stem = f"{file_stem}_log_scale_pca"
print(file_stem)

# Apply log to the features (because the statement recommends it).
# Leave zeroes intact.
X = np.ma.log(X.to_numpy()).filled(0)

# Normalize features.
X = StandardScaler().fit_transform(X)

# Apply PCA.
X = PCA().fit_transform(X)

# Build PC dataframe.
pc_data_frame = pd.DataFrame(
    data=X,
    columns=[f"pc_{idx}" for idx in range(1, X.shape[1] + 1)],
)
pc_data_frame = pd.concat([pc_data_frame, data_frame.loc[:, klass_cols]], axis=1)

# Graph PCA.
for klass_name in klass_cols:
    klass_list = data_frame[klass_name].unique()
    color_list = list(mcolors.TABLEAU_COLORS.keys())
    x_label = "pc_1"
    y_label = "pc_2"

    for klass, color in zip(klass_list, color_list):
        idx_list = pc_data_frame[klass_name] == klass
        plt.scatter(
            pc_data_frame.loc[idx_list, x_label],
            pc_data_frame.loc[idx_list, y_label],
            marker="o",
            color=color,
        )

    plt.title(f"{file_stem} with target {klass_name}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(klass_list)
    plt.savefig(fname=f"{file_stem}_{klass_name}")

# Convert classes to integers.
for klass in klass_cols:
    pc_data_frame[klass] = pd.factorize(pc_data_frame[klass])[0]

# Export data.
pc_data_frame.to_csv(f"{file_stem}.csv")
