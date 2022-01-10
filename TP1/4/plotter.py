import matplotlib.pyplot as plt

# Read from file.
input_file = open(f"wine_rfe_svm.err", "r")
feature_count = []
test_error = []
for _ in range(13):
    feature_cnt, test_err = [
        float(value) for value in input_file.readline().split()
    ]
    feature_count.append(feature_cnt)
    test_error.append(test_err)

# Plot graph.
plt.plot(
    feature_count,
    test_error,
    marker="o",
    color="tab:blue"
)
plt.title(f"SVM classifier for dataset wine")
plt.xlabel("Amount of variables")
plt.ylabel("Error rate")
plt.savefig(fname=f"RFE_SVM_wine_graph")
