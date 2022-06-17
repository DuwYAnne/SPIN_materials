import numpy as np
from pymatgen.ext.matproj import MPRester
import pymatgen.core as mg
import csv

# Generates 1000 material id's and writes to csv file
class Generator():
    def __init__(self, size=126335):
        MAPI = "RDq2I57WXBqDBeTtPz" # use API key from materialproject.org
        self.mpr = MPRester(MAPI)
        self.bandgap_data = self.mpr.query({"band_gap": {"$exists": True}},
                        properties=["task_id", "band_gap"])[:size]
    def get_bandgap_pairs(self):
        id_prop = []
        for item in self.bandgap_data:
            id = int(''.join(filter(str.isdigit, item['task_id'])))
            id_prop.append((id, item['band_gap']))
        return id_prop
    def data_as_dict(self):
        structures = {}
        for item in self.bandgap_data:
            id = int(''.join(filter(str.isdigit, item['task_id'])))
            structure = self.mpr.get_structure_by_material_id(item['task_id'])
            structures[id] = structure
        print("DONE STRUCTURES")
        return structures
    
    def write_id_prop(self, path):
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            for item in self.bandgap_data:
                id = int(''.join(filter(str.isdigit, item['task_id'])))
                writer.writerow([id, item['band_gap']])

        # Generates csv file with dict {idx, structure.as_dict} pairs
    def write_id_struct(self, path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            for item in self.bandgap_data:
                structure = self.mpr.get_structure_by_material_id(item['task_id'])
                id = int(''.join(filter(str.isdigit, item['task_id'])))
                writer.writerow([id, structure.to(fmt='cif')])
    