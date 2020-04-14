import numpy as np
from os.path import join
from itertools import product
from joblib import Parallel, delayed
import os
DATA_PATH = './data'
NUM_ORBIT = 98304
NUM_THREADS = 6

def filename(path, kind, ind):
    return join(path, '{}_{}.txt'.format(kind, ind))

def save_all(p, kind):
    path = join(DATA_PATH, p)
    if kind in ('input', 'output'):
        all_data = [np.loadtxt(filename(path, kind, ind)) for ind in range(0, NUM_ORBIT)]
        all_data = np.concatenate(all_data, axis=0)
        np.savetxt(filename(DATA_PATH, kind, '{}_concatenated'.format(p)), all_data)

if __name__ == "__main__":
    path = ['train', 'validation', 'test']
    kind = ['input', 'output']
    Parallel(n_jobs=NUM_THREADS, verbose=3)([delayed(save_all)(p, k) for p, k in product(path, kind)])
