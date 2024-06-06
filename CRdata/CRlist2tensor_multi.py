import pickle
import torch
import os


directory = "lists"
pkl_files = [file for file in os.listdir(directory) if file.endswith(".pkl")]

for file_path in pkl_files:
    with open(f"{directory}/{file_path}", "rb") as file:
        data = pickle.load(file)
    data = data[500:]
    data = torch.tensor(data, dtype=torch.long)

    with open(f"tmp/tensor_{file_path}", "wb") as file:
            pickle.dump(data, file)

