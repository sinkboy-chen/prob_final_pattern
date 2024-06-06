import pickle
import torch

file_path = "cr_g4_child9.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)

data = data[500:]
data = torch.tensor(data, dtype=torch.long)
# print(len(data))

# print(len(data))
# data = torch.tensor(data, dtype=torch.long)

# print(len(data))
# print(type(data))
# print(type(data[0]))

with open("gcd_cr_tensor.pkl", "wb") as file:
        pickle.dump(data, file)

