from re import X
from pymatgen.core.structure import Structure
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from utils import GraphConstructor
import random
import math
from itertools import product
crystal = pyxtal()
test = [list(elem) for elem in product(range(0, 1), repeat=2)]
sum_pos = []
for i in range(1, 231):
    g = Group(i)
    total_sum = 0
    for position in g.Wyckoff_positions:
        total_sum += position.multiplicity
    sum_pos.append(total_sum)
print(sum_pos)
dataset = []
gc = GraphConstructor()
for i in range(1, 231):
    g = Group(i)
    total = 0
    curr = []
    for natoms in range(1, 5):
        print(i, natoms)
        test = [list(elem) for elem in product(range(1, 20 // natoms), repeat=natoms)]
        for item in test:
            if math.gcd(*item) != 1:
                continue
            else:
                total += 1
                crystal.from_random(3, i, random.sample(range(3, 80), natoms), item)
                #curr.append(gc.structure_to_graph(crystal.to_pymatgen(), i))
    #for item in random.sample(curr, 10):
    #    dataset.append(item)
                