import sys
import numpy as np

if len(sys.argv) != 4:
    raise Exception("Usage: uniform_noise.py file_stem feat_dim noise_dim")

file_stem = str(sys.argv[1])
feat_dim = int(sys.argv[2])
noise_dim = int(sys.argv[3])

input_file = open(f"{file_stem}.data", "r")
output_file = open(f"{file_stem}_noise.data", "w")
rng = np.random.default_rng()

output_line = [f"V{index + 1}" for index in range(feat_dim + noise_dim)] + ["y"]
output_file.write(str(output_line)[1:-1].replace(",", "").replace("'", '"') + "\n")

for str_line in input_file.readlines():
    input_line = list(map(float, str_line.split(",")))
    features = np.array(input_line[:-1])
    klass = int(input_line[-1])
    noise = rng.random(noise_dim) * 2 - 1
    output_line = np.concatenate([features, noise])
    output_file.write(" ".join(map(str, output_line)) + f' "{klass}"\n')
