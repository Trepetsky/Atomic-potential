import ase
from ase.db import connect
import numpy as np
from joblib import Parallel
from joblib import delayed
import itertools
from time import perf_counter

start = perf_counter()

db = connect(r'drive/MyDrive/APNN_ethanol.db')
number_molecules = len(db)

H_description = []
O_description = []
C_description = []
data_energy = []

row_list = []
for row in db.select():
    row_list.append(row)


def descriptor(row):
    atoms = row.toatoms()
    change_nuclear = atoms.get_atomic_numbers()
    position_atoms = atoms.get_positions()
    number_atoms = len(change_nuclear)
    change_nuclear_X_distance_list = []
    for i in range(number_atoms):
        for n in range(number_atoms):
            if i != n:
                distance = (sum(position_atoms[i] - position_atoms[n]) ** 2) ** (1 / 2)
                change_nuclear_X_distance_list.append(distance * change_nuclear[n])
    change_nuclear_X_distance_list = np.array(change_nuclear_X_distance_list).reshape(number_atoms, number_atoms - 1)
    descriptor = 0.33 * np.log(np.sum(change_nuclear_X_distance_list ** 101, axis=1))
    H_description_molecule_N = []
    O_description_molecule_N = []
    C_description_molecule_N = []
    if 1 not in change_nuclear:
        H_description_molecule_N.append(np.nan)
    else:
        for i in range(number_atoms):
            if change_nuclear[i] == 1:
                H_description_molecule_N.append(descriptor[i])
    if 8 not in change_nuclear:
        O_description_molecule_N.append(np.nan)
    else:
        for i in range(number_atoms):
            if change_nuclear[i] == 8:
                O_description_molecule_N.append(descriptor[i])
    if 6 not in change_nuclear:
        C_description_molecule_N.append(np.nan)
    else:
        for i in range(number_atoms):
            if change_nuclear[i] == 6:
                C_description_molecule_N.append(descriptor[i])
    return [H_description_molecule_N, O_description_molecule_N, C_description_molecule_N, row.data.energy[0]]


data = Parallel(n_jobs=-1)(delayed(descriptor)(row) for row in row_list)
data = list(itertools.chain(*data))
for i in range(0, len(data), 4):
  H_description.append(data[i])
for i in range(1, len(data), 4):
  O_description.append(data[i])
for i in range(2, len(data), 4):
  C_description.append(data[i])
for i in range(3, len(data), 4):
  data_energy.append(data[i])
end = perf_counter()
print((end - start) / 60, "мин.")

length = max(map(len, H_description))
H_description = np.array([i + [np.nan]*(length-len(i)) for i in H_description])

length = max(map(len, O_description))
O_description = np.array([i + [np.nan]*(length-len(i)) for i in O_description])

length = max(map(len, C_description))
C_description = np.array([i + [np.nan]*(length-len(i)) for i in C_description])

data_energy = np.array(data_energy)

np.savez("data_array", H_description=H_description, O_description=O_description, C_description=C_description, data_energy=data_energy)
