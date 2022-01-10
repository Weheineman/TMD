import sys
import subprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from variable_selection.wrapper import ForwardWrapper, BackwardWrapper
from variable_selection.filter import KruskalWallis
from variable_selection.embedded import RecursiveFeatureElimination
from variable_selection.data import Data


def count_hits(list1, list2):
    """Returns the amount of unique common elements between both lists."""
    hit_set = set(list1).intersection(list2)
    return len(hit_set)


repetitions = 30
file_stem = "diagonal_noise"
feat_dim = 10
feat_names = [f"V{num}" for num in range(1, feat_dim + 1)]
wrapper_rounds = 5

score = {
    "Forward Wrapper SVM": 0,
    "Backward Wrapper SVM": 0,
    "Kruskal Wallis": 0,
    "Recursive Feature Elimination SVM": 0,
    "Recursive Feature Elimination RF": 0,
}

for rep in range(repetitions):
    subprocess.run(["python", "diagonal.py", "100", "10", "2"])
    subprocess.run(["python", "uniform_noise.py", "diagonal", "10", "90"])

    # Read data.
    data = Data(file_stem)
    data.read_train_data()

    # # Run the forward wrapper with SVM.
    svm_fwrapper = ForwardWrapper(SVC(), data)
    svm_fwrapper.select_variables()

    # # Run the backward wrapper with SVM.
    svm_bwrapper = BackwardWrapper(SVC(), data)
    svm_bwrapper.select_variables()

    # Run the filter.
    kruskal_wallis = KruskalWallis(data)
    kruskal_wallis.filter_variables(feat_dim)

    # Run the RFE with SVM.
    svm_rfe = RecursiveFeatureElimination(SVC(kernel="linear"), data, "coef_")
    svm_rfe.rank_variables(1)

    # Run the RFE with Random Forest.
    rf_rfe = RecursiveFeatureElimination(
        RandomForestClassifier(), data, "feature_importances_"
    )
    rf_rfe.rank_variables(1)

    # Add the hits.
    score["Forward Wrapper SVM"] += count_hits(
        svm_fwrapper.ranking[:feat_dim], feat_names
    )
    score["Backward Wrapper SVM"] += count_hits(
        svm_bwrapper.ranking[:feat_dim], feat_names
    )
    score["Kruskal Wallis"] += count_hits(
        kruskal_wallis.filtered_variables(), feat_names
    )
    score["Recursive Feature Elimination SVM"] += count_hits(
        svm_rfe.ranking[:feat_dim], feat_names
    )
    score["Recursive Feature Elimination RF"] += count_hits(
        rf_rfe.ranking[:feat_dim], feat_names
    )


for method, hits in score.items():
    hit_rate = hits / repetitions / feat_dim * 100
    print(f"{method}: {hits}/{repetitions * feat_dim} = {hit_rate}%")
