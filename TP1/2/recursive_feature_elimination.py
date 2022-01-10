import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from variable_selection.embedded import RecursiveFeatureElimination
from variable_selection.data import Data

file_stem = sys.argv[1]
print(f"RFE usando el dataset {file_stem}.")

# Read data.
data = Data(file_stem)
data.read_train_data()

execution_count = 5

# Run the RFE with SVM.
print(f"Resultado de {execution_count} ejecuciones usando SVM:")
for _ in range(execution_count):
    svm_rfe = RecursiveFeatureElimination(SVC(kernel="linear"), data, "coef_")
    svm_rfe.rank_variables(1)
    print(svm_rfe.ranking)

# Run the RFE with Random Forest.
print(f"Resultado de {execution_count} ejecuciones usando Random Forest:")
for _ in range(execution_count):
    rf_rfe = RecursiveFeatureElimination(
        RandomForestClassifier(), data, "feature_importances_"
    )
    rf_rfe.rank_variables(1)
    print(rf_rfe.ranking)
