import pickle
import torch

file_path = "gcd_g4_child13.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)

data = torch.tensor(data, dtype=torch.long)

with open("gcd_ga_tensor.pkl", "wb") as file:
        pickle.dump(data, file)

