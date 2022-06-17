import sys
sys.path.insert(0, 'C:/users/maxwe/SPIN_materials')
from pymatgen.core.structure import Structure
from matminer.datasets import load_dataset
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import gzip
import matplotlib as plt
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph

with open('./datasets/matbench_sample_10000_mp_gap.json', 'r') as f:
    df = pd.read_json(f.read(), orient='index')
df['structure'] = df['structure'].apply(Structure.from_dict)
targets = list(df['gap'].values)
structures = list(df['structure'])

nfeat_bond = 10
r_cutoff = 6.0
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
model.train(structures, targets, epochs=100)
model.save_model('./saved-models/megnet-10000-gap.hdf5')