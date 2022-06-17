import sys
sys.path.insert(0, 'C:/users/maxwe/SPIN_materials')
import numpy as np
import numpy.linalg as la
import pymatgen.core as mg
import json
import time
from pymatgen.core.structure import Structure

import importlib
import utils
from utils import GraphConstructor
import pandas as pd
import cython
import csv
import torch
from random import randint
import ocpmodels
from ocpmodels.trainers import RegressionTrainer
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging
setup_logging()
with open('./datasets/matbench_sample_1000_mp_e_form.json', 'r') as f:
    df = pd.read_json(f.read(), orient='index')
df['structure'] = df['structure'].apply(Structure.from_dict)
targets = list(df['e_form'].values)
structures = list(df['structure'])
start = time.time()
gc = GraphConstructor()
dataset = []
idx = 0
for crystal in structures:
    dataset.append(gc.structure_to_graph(crystal, target=targets[idx], sid=idx))
    #dataset[-1].y = torch.FloatTensor(targets[idx])
    idx += 1
end = time.time()
print(end-start)
#print(dataset)

dataset_in = {
    'train': dataset[:90],
    'val': dataset[90:],
    #'test': dataset[90:],
    'normalizer_info': {
        "normalize_labels": False,
    }
}
task = {
    'dataset': 'single_point_lmdb',
    'description': 'Regressing to energies and forces for DFT trajectories from OCP',
    'type': 'regression',
    'metric': 'mae',
    'labels': ['formation energy'],
    'train_on_free_atoms': True,
    'eval_on_free_atoms': True
}

model = {
    'name': 'painn',
    'hidden_channels': 1024,
    'num_layers': 3,
    'num_rbf': 128,
    'cutoff': 6.0,
    'max_neighbors': 50,
    'regress_forces': False,
    'use_pbc': True
}

optim = {
    'batch_size': 64,
    'eval_batch_size': 32,
    'load_balancing': 'atoms',
    'num_workers': 1,
    'optimizer': 'AdamW',
    'optimizer_params': {"amsgrad": True},
    'lr_initial': 1.e-4,
    'scheduler': 'ReduceLROnPlateau',
    'mode': 'min',
    'factor': 0.8,
    'patience': 3,
    'max_epochs': 20,
    'energy_coefficient': 1,
    'ema_decay': 0.999,
    'clip_grad_norm': 1,
    'weight_decay': 0.0
}

trainer = RegressionTrainer(
    task=task,
    model=model,
    dataset=dataset_in,
    optimizer=optim,
    identifier='painn-test',
    run_dir="./",
    is_debug=False,
    cpu=True,
    print_every=5,
    seed=0,
    logger="tensorboard",
    local_rank=0,
    amp=False
)
trainer.train()