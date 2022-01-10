from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from variable_selection.embedded import RecursiveFeatureElimination
from variable_selection.data import Data


# Get data.
data = Data()
values, results = load_wine(return_X_y=True)

# Separate a balanced test sample.
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sss.split(values, results):
    data.train_values = values[train_index]
    data.train_results = results[train_index]
    data.test_values = values[test_index]
    data.test_results = results[test_index]

# Relate names to indexes.
data.variable_names = load_wine().feature_names
name_to_idx = {}
for idx, name in enumerate(data.variable_names):
    name_to_idx[name] = idx

# Run the RFE with SVM.
svm_rfe = RecursiveFeatureElimination(SVC(kernel="linear"), data, "coef_")
svm_rfe.rank_variables(1)

output_file = open("wine_rfe_svm.err", "w")

for var_count in range(1,len(data.variable_names) + 1):
    indexes = [name_to_idx[name] for name in svm_rfe.ranking[:var_count]]
    train_values = [data_point[indexes] for data_point in data.train_values]
    svc = SVC()
    svc.fit(train_values, data.train_results)
    test_values = [data_point[indexes] for data_point in data.test_values]
    predictions = svc.predict(test_values)
    error_rate = sum(
        [
            prediction != result
            for prediction, result in zip(predictions, data.test_results)
        ]
    ) / len(predictions)
    output_file.write(f"{var_count} {error_rate}\n")
