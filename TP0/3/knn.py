from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import sys

# Read data.
file_stem = sys.argv[1]
train_file = open(f"{file_stem}.data", "r")
test_file = open(f"{file_stem}.test", "r")
train_values = []
train_classes = []
test_values = []
test_classes = []

# Parse train and test data.
for file, values, classes in zip(
    [train_file, test_file], [train_values, test_values], [train_classes, test_classes]
):
    for line in file.readlines():
        input_list = [float(value) for value in line.split(",")]
        values.append(input_list[:-1])
        classes.append(str(input_list[-1]))

# Fit parameters for test data.
knn = KNeighborsClassifier()
parameters = {"n_neighbors": range(1, 11)}
knn = GridSearchCV(knn, parameters, cv=5)  # 5-fold cross-validation.
knn.fit(train_values, train_classes)

# Print chosen parameters.
print(f"Parametros escogidos: {knn.best_params_}")

# Print training score (estimated error).
print(f"Error estimado de test (5-fold CV): {100 * (1 - knn.best_score_)}%")

# Print test score.
print(f"Error de test: {100 * (1 - knn.score(test_values, test_classes))}%")
