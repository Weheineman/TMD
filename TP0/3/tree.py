from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
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

tree = DecisionTreeClassifier()

# Print training score (estimated error).
estimated_score = cross_val_score(tree, train_values, train_classes, cv=5).mean()
print(f"Error estimado de test (5-fold CV): {100 * (1 - estimated_score)}%")

# Fit parameters for test data.
tree.fit(train_values, train_classes)

# Print test score.
print(f"Error de test: {100 * (1 - tree.score(test_values, test_classes))}%")
