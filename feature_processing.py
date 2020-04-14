import numpy as np
from os import path
import os
from joblib import Parallel, delayed

NUM_ORBIT = 98304
NUM_THREADS = os.cpu_count() // 2
DATA_DIR = "./data"
SCALER = 20.0

def process_input(data):
    #r_inv = 1. / np.sqrt(np.square(data[:, :2]).sum(axis=1)).reshape(-1, 1)
    #return np.concatenate([data[:, :4], r_inv], axis=1)[:-1, :]
    return data[:-1, :4] / SCALER

def process_output(data):
    return (data[1:, :4] - data[:-1, :4]) / SCALER

def get_file_data(dir_name, ind):
    file_name = str(ind) + ".txt"
    return np.loadtxt(path.join(dir_name, file_name))

def save_data(data, dir_name, ind, file_type):
    file_name = file_type + '_' + str(ind) + ".txt"
    np.savetxt(path.join(dir_name, file_name), data)

def produce_input_output_file(dir_name, ind):
    raw_data = get_file_data(dir_name, ind)
    input_data = process_input(raw_data)
    save_data(input_data, dir_name, ind, 'input')
    
    output_data = process_output(raw_data)
    save_data(output_data, dir_name, ind, 'output')

def multi_process(ind):
    produce_input_output_file(path.join(DATA_DIR, "train"), ind)
    produce_input_output_file(path.join(DATA_DIR, "validation"), ind)
    produce_input_output_file(path.join(DATA_DIR, "test"), ind)


if __name__ == "__main__":
    Parallel(n_jobs=NUM_THREADS, verbose=3)([delayed(multi_process)(ind) for ind in range(NUM_ORBIT)]) 
