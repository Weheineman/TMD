import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from variable_selection.wrapper import ForwardWrapper
from variable_selection.data import Data

file_stem = sys.argv[1]
print(f"Forward Wrapper usando el dataset {file_stem}.")

# Read data.
data = Data(file_stem)
data.read_train_data()

execution_count = 5

# Run the wrapper with SVM.
print(f"Resultado de {execution_count} ejecuciones usando SVM:")
for _ in range(execution_count):
    svm_wrapper = ForwardWrapper(SVC(), data)
    svm_wrapper.select_variables()
    print(svm_wrapper.selected_variables())

# Run the wrapper with Random Forest.
print(f"Resultado de {execution_count} ejecuciones usando Random Forest:")
for _ in range(execution_count):
    svm_wrapper = ForwardWrapper(RandomForestClassifier(), data)
    svm_wrapper.select_variables()
    print(svm_wrapper.selected_variables())
