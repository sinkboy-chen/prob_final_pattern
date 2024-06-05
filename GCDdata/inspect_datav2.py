import pickle
import numpy as np
import math

def count_answer(data):
    count = dict()
    total = 0
    for row in data:
        answer = row[4]*10+row[5]
        if answer not in count:
            count[answer] = 1
        else:
            count[answer] += 1
        total += 1
    sorted_dict = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
    print(total)
    return sorted_dict

file_path = "tmp/gcd_ga_tensor.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)

data = data.numpy()

for i in range(math.ceil(len(data)/64)):
    count_answer(data[i*64:min((i+1)*64, len(data))])

