import pickle
import numpy as np
import random
import torch

def get_answer(row):
    answer = row[4]*10+row[5]
    return answer

def find_row(data, min_index, max_answer):
    for _ in range(100):
        index = random.randint(min_index, len(data)-1)
        if get_answer(data[index])<=max_answer:
            return index
    for index in range(min_index, len(data)):
        if get_answer(data[index])<=max_answer:
            return index
    assert 0
        
def make_phase(data, max_answer, lower_index, upper_index):
    for i in range(lower_index, upper_index):
        answer = get_answer(data[i])
        if answer>max_answer:
            swap_index = find_row(data, upper_index, max_answer)
            data[[i, swap_index]] = data[[swap_index, i]]
    np.random.shuffle(data[lower_index:upper_index])

def count_answer(data, count={}):
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

def make_data(save_path):
    print(f"making {save_path}")
    file_path = "init.pkl"
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    data = data.numpy()
    np.random.shuffle(data)
    # first 20 batch contains 1-5
    make_phase(data, 5, 0, 64*20)
    # middle 21-40 batch contains 1-10
    make_phase(data, 10, 64*20, 64*40)
    make_phase(data, 23, 64*41, 64*67)

    # # check it is correct
    # count = count_answer(data[:20*64])
    # count = count_answer(data[20*64:40*64], count)
    # count = count_answer(data[40*64:], count)

    data = torch.tensor(data)

    with open(save_path, "wb") as file:
        pickle.dump(data, file)

for i in range(21, 30):
    make_data(f"curriculum/{i}.pkl")



