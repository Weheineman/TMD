import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

file_stem = "diag_train"
# Read data.
data_frame = pd.read_csv(f"{file_stem}.csv")

color_list = list(mcolors.TABLEAU_COLORS.keys())
for idx, klass in enumerate(data_frame["class"].unique()):
    idx_list = data_frame["class"] == klass
    plt.scatter(
        data_frame.loc[idx_list, "V1"],
        data_frame.loc[idx_list, "V2"],
        marker="o",
        color=color_list[idx],
    )
plt.savefig(f"{file_stem}")