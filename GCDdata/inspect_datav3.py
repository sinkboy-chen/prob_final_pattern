import pickle
import numpy as np
import torch
import random

def shuffle_tensor(num_shuffle, data):
    data = data.numpy()
    data = list(data)

    # process on data, no return value
    num_data = len(data)
    assert(num_data>=num_shuffle)
    shuffle_index = random.sample(range(num_data), num_shuffle)
    shuffle_value = []
    for index in shuffle_index:
        shuffle_value.append(data[index])
    random.shuffle(shuffle_value)
    for i in range(num_shuffle):
        data[shuffle_index[i]] = shuffle_value[i]

    data = np.array(data)
    data = torch.tensor(data)
    return data

file_path = "init.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)

print(data[:10])
data = data.numpy()
data = list(data)
data = np.array(data)
data = torch.tensor(data)

data = shuffle_tensor(4351, data)
print(data[:10])