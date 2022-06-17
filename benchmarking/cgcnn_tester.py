import sys
sys.path.insert(0, 'C:/users/maxwe/SPIN_materials')
import cgcnn.train as runner
from cgcnn.cgcnn.data import CIFData_direct
from pymatgen.core.structure import Structure
from matminer.datasets import load_dataset
import numpy as np
import pandas as pd
import json
import gzip
from getCIF import Generator
import matplotlib as plt
''' 

Code to train cgcnn model from materials project datasets


#Let our learning rate vary
gen = Generator(10)
#print(gen.data_as_dict())
dict_data = gen.data_as_dict()
id_prop = gen.get_bandgap_pairs()
dataset = CIFData_direct('./cgcnn/cgcnn/data/atom_init.json', 
                        dict_data, id_prop)
tests = []
for lr in [.01, .001, 3e-5]:
    for test_ratio in [.1, .2, .3]:
        for n_h in [1, 3, 5]:
            for optim in ['SGD', 'Adam']:
                mae = runner.test(dataset, lr=lr, test_ratio=test_ratio, n_h=n_h, optim=optim
                                ,epochs=12, print_freq=10).tolist()
                tests.append({'lr':lr, 'test-ratio':test_ratio,
                'n_h':n_h, 'optim':optim, 'mae':mae})
with open('./hyperparam_search.json', 'w') as f:
    json.dump(tests, f)
'''

with open('./datasets/matbench_sample_10000_mp_e_form.json', 'r') as f:
    df = pd.read_json(f.read(), orient='index')
df['structure'] = df['structure'].apply(Structure.from_dict)
dict_data = df['structure'].to_dict()
id_prop = list(zip(list(df.index.values), (df['e_form'].values)))
dataset = CIFData_direct('./cgcnn/cgcnn/data/atom_init.json', 
                        dict_data, id_prop)
lr = .001
n_h = 3
optim = 'Adam'
mae = runner.test(dataset, lr=lr,  n_h=n_h, optim=optim
                    ,epochs=100, print_freq=10, save_path='./saved-models/cgcnn-10000-e-form.pth').tolist()

tests = []
tests.append({'lr':lr, 'n_h':n_h, 'optim':optim, 'mae':mae})
with open('./matbench1.json', 'w') as f:
    json.dump(tests, f)
