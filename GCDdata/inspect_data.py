import pickle
import numpy as np

file_path = "init.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)

count = {}
total = 0
data = data.numpy()
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