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

def count_answer(data, count):
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
    data = list(data)
    data.sort(key=get_answer)
    # for row in data:
    #     print(get_answer(row))
    data = np.array(data)
    # print(data)
    data = torch.tensor(data)

    with open(save_path, "wb") as file:
        pickle.dump(data, file)


make_data(f"sorted.pkl")



