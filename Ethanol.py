""" То над чем сейчас работаю, код пока не работает, в процессе написания"""
import os
import numpy as np
import ase
from ase.io import read
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

if not os.path.exists('./ethanol_dft.zip'):
    !wget http://quantum-machine.org/gdml/data/xyz/ethanol_dft.zip

if not os.path.exists('./ethanol.xyz'):
    !unzip ./ethanol_dft.zip

ethanol = read('./ethanol.xyz', index=':10')
number_molecules = len(ethanol)
description_list = []
property_list = []
for at in range(number_molecules):
    # All properties need to be stored as numpy arrays.
    # Note: The shape for scalars should be (1,), not ()
    # Note: GPUs work best with float32 data
    change_nuclear = ethanol[at].get_atomic_numbers()
    position_atoms = ethanol[at].get_positions()
    number_atoms = len(change_nuclear)
    change_nuclear_X_distance_list = []
    for i in range(number_atoms):
      for n in range(number_atoms):
        if i != n:
          distance = (sum(position_atoms[i] - position_atoms[n])**2)**(1/2)
          change_nuclear_X_distance_list.append(distance*change_nuclear[n])
    change_nuclear_X_distance_list = np.array(change_nuclear_X_distance_list).reshape(number_atoms,number_atoms-1)
    descriptor = 0.33*np.log(np.sum(change_nuclear_X_distance_list**101, axis=1))
    for i in range(number_atoms):
      if change_nuclear[i] == 6:
        description_list.append({"C": descriptor[i]})
      elif change_nuclear[i] == 8:
        description_list.append({"O": descriptor[i]})
      elif change_nuclear[i] == 1:
        description_list.append({"H": descriptor[i]})
    energy = np.array([float(list(ethanol[at].info.keys())[0])], dtype=np.float64)
    property_list.append(
        {'energy': energy}
    )
description_list = np.array(description_list).reshape(number_molecules, number_atoms)

H_description = []
O_description = []
C_description = []
data_energy = []
for molecule_N in description_list:
  for n in range(number_atoms):
    H_description.append(molecule_N[n].get('H'))
  for n in range(number_atoms):
    O_description.append(molecule_N[n].get('O'))
  for n in range(number_atoms):
    C_description.append(molecule_N[n].get('C'))
for energy_N in property_list:
  data_energy.append(energy_N['energy'][0])

H_description = np.array(H_description, dtype=np.float64).reshape(number_molecules, number_atoms)
O_description = np.array(O_description, dtype=np.float64).reshape(number_molecules, number_atoms)
C_description = np.array(C_description, dtype=np.float64).reshape(number_molecules, number_atoms)
data_energy = np.array(data_energy).reshape(-1, 1)


n_cols = 1
input1 = keras.Input(shape=(n_cols,), name="H_element")
l1_1 = Dense(8, activation='relu')(input1)
l1_2 = Dense(1, activation='linear',name = 'H_atomic_energy')(l1_1) #atomic energy for H

# graph for O
input2 = keras.Input(shape=(n_cols,), name="O_element")
l2_1 = Dense(8, activation='relu')(input2)
l2_2 = Dense(1, activation='linear', name = 'O_atomic_energy')(l2_1) #atomic energy for O

input3 = keras.Input(shape=(n_cols,), name="C_element")
l3_1 = Dense(8, activation='relu')(input3)
l3_2 = Dense(1, activation='linear', name = 'O_atomic_energy')(l3_1) #atomic energy for O

# summation and output. Total energy is the target
x = layers.concatenate([l1_2, l2_2, l3_2])
#ReduceSum = Lambda(lambda z: K.sum(z, axis=1, keepdims=True))
def ReduceSum(z):
    return K.sum(z, axis=1, keepdims=True)
output = ReduceSum(x)

model = keras.Model(
    inputs=[input1, input2, input3],
    outputs=output,)

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

history = model.fit(c, f, epochs=500, verbose=0)
print("Обучение завершено")
x=1
y=1
#print(model.predict([x, y]))
print(model.get_weights())

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
