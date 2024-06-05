# evaluate all the data.pkl in a folder and save it to folder/log.txt
# iteration, pkl_name, evaluation time, machine

import os
import sys
import json
import numpy as np
import random
import nltk
np.set_printoptions(threshold=np.inf)

import torch
# from ChickenRabbit import ChickenRabbitDataset, eval_split
from GCD import GCDDataset, eval_split
from torch.utils.data.dataloader import DataLoader
torch.set_printoptions(profile="full")

from mingpt.model_multiplier import GPT
from mingpt.trainer_multiplier import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from itertools import permutations

import socket
import time
import pickle

# -----------------------------------------------------------------------------

def get_config():
    C = CN()

    # system
    C.system = CN()
    # TODO: random seed for model can be set here
    C.system.init_seed = 0 # will change the weight initialization
    C.system.work_dir = './test'

    # data
    C.data = GCDDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    
    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.task = "gcd" # or gcd
    return C

def batch_end_callback(trainer, model, train_dataset, test_dataset):
    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    if trainer.iter_num % 50 == 0:
        # evaluate both the train and test acc
        model.eval()
        with torch.no_grad():
            train_mean = eval_split(trainer.device, model, train_dataset)
            test_mean  = eval_split(trainer.device, model, test_dataset)
        print(f'the mean of train and test are {train_mean}, {test_mean}')
        # save the model and terminate the training
        if test_mean >= 0.9:
            print(f"reach threshold 0.9 in iteration: {trainer.iter_num}")
            print(f"saving model with test_mean: {test_mean}")
            ckpt_path = os.path.join(f"test/{trainer.config.task}", "model_last.pt")
            torch.save(model.state_dict(), ckpt_path)
            return trainer.iter_num
        # revert model to training mode
        model.train()
    return -1

def write_log(directory, message):
    with open(f"{directory}/log.txt", "a") as file:
        file.write(f"{message}\n")
        print(f"{message}")

def evaluate_train_data(train_data_path):
    start = time.time()
    config = get_config()
    setup_logging(config)

    # TODO: try different seed for model
    set_seed(config.system.init_seed)

    # TODO: try different seed to adjust the data order of train/test-set
    train_dataset = GCDDataset(config.data, split='train', seed=0, defined_tensor_train_data=train_data_path)
    test_dataset  = GCDDataset(config.data, split='test', seed=0)

    # set the correct vocab size: 10, block size: chickenrabbit -> 10, gcd -> 6
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    stop_iteration = trainer.run()
    if stop_iteration != -1:
        print(f'The final iteration of this round is {stop_iteration}!')
    else:
        print('It cannot reach 0.9 acc within max_iteration steps...')
    
    end = time.time()
    return stop_iteration, end-start

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

def mutate(original_data_path, new_data_path, num_shuffle):
    with open(original_data_path, "rb") as file:
        data = pickle.load(file)
    data = shuffle_tensor(num_shuffle, data)
    with open(new_data_path, "wb") as file:
        pickle.dump(data, file)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    original_data_path = "GCDdata/init.pkl"
    directory = "GCDdata/ga/v1"
    best_iteration = 25000

    for i in range(100, 0, -1):
        new_data_path = f"{directory}/{i}.pkl"
        mutate(original_data_path, new_data_path, 20)
        write_log(directory, f"{new_data_path} made.")
        write_log(directory, f"evaluating {new_data_path}")
        stop_iteration, duration = evaluate_train_data(new_data_path)
        write_log(
                directory,
                f"iteration: {stop_iteration}, train: {new_data_path}, duration: {duration}, host: {socket.gethostname()}, gpu: {os.getenv('CUDA_VISIBLE_DEVICES')}"
                )
        if stop_iteration!=-1 and stop_iteration<=best_iteration:
            original_data_path = new_data_path
            best_iteration = stop_iteration
            write_log(directory, "")
            write_log(directory, f"best iteration: {best_iteration}, train: {original_data_path}")
            write_log(directory, "")


    


    
