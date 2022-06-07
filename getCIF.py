import numpy as np
from pymatgen.ext.matproj import MPRester
import pymatgen.core as mg
MP_ID = "mp-19017"
MAPI = "RDq2I57WXBqDBeTtPz" # use API key from materialproject.org
mpr = MPRester(MAPI)
structure = mpr.get_structure_by_material_id(MP_ID)
bs = mpr.get_bandstructure_by_material_id(MP_ID)
print(bs.get_band_gap()['energy'])
print(structure.to(fmt='cif'))
