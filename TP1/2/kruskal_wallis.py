import sys

from variable_selection.filter import KruskalWallis
from variable_selection.data import Data

file_stem = sys.argv[1]
print(f"Kruskal-Wallis usando el dataset {file_stem}.")

remaining_amount = int(sys.argv[2])

# Read data.
data = Data(file_stem)
data.read_train_data()

# Run the filter.
kruskal_wallis = KruskalWallis(data)
kruskal_wallis.filter_variables(remaining_amount)

# Print the result.
print(f"Resultado de elegir las mejores {remaining_amount} variables:")
print(kruskal_wallis.filtered_variables())