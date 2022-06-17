import numpy as np
import numpy.linalg as la
import pymatgen.core as mg
import json
import gzip
from pymatgen.core.structure import IStructure
from pymatgen.core.structure import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
import pandas as pd
from pymatgen.ext.matproj import MPRester
import csv
from random import randint

MAPI = "RDq2I57WXBqDBeTtPz"
mpr = MPRester(MAPI)
#with open('./datasets/matbench_sample_1000_mp_e_form.json', 'r') as fin:
#    json_str = fin.read()
#df = pd.read_json(json_str, orient="index")
#test = Structure.from_dict(df['structure'].iloc[0])

#df = load_dataset("matbench_mp_gap")
#with gzip.open('./datasets/matbench_mp_gap.json.gz', 'r') as fin:
#    df = pd.read_json(fin.read().decode('utf-8'), orient="index")
#df = df.T
#print(df)
#df[['structure', 'gap']] = pd.DataFrame(df.data.tolist(), index=df.index)
#df = df[['structure', 'gap']]
#df = df.sample(n=100)
#print(df)
#data = df.to_json('./datasets/matbench_sample_100_mp_gap.json', orient="index")

#with open('./datasets/matbench_sample_100_mp_e_form.json', 'r') as f:
#    df = pd.read_json(f.read(), orient='index')
#df['structure'] = df['structure'].apply(Structure.from_dict)
#structs = list(df['structure'])
#print(structs)
test = mpr.get_structure_by_material_id("mp-20470")
#test = structs[0]
print(test)
lattice_matrix = np.array(test.lattice.matrix)
print(lattice_matrix)
cart_coords = np.array(test.cart_coords)
print(cart_coords)
tol: float=1e-8
center_indices, neighbor_indices, images, distances = find_points_in_spheres(
    cart_coords, cart_coords, r=3.0, pbc=np.array([1,1,1], dtype=int), lattice=lattice_matrix, tol=tol
)
print(center_indices)
print(neighbor_indices)
print(images)
images = np.array(images)
#print(images[0].reshape(1, 3).shape)
print(distances)
