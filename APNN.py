import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from time import perf_counter
import matplotlib.pyplot as plt

data_array = np.load("data_array.npz")
H_description = data_array["H_description"]
O_description = data_array["O_description"]
C_description = data_array["C_description"]
data_energy = data_array["data_energy"]

perm = np.random.permutation(H_description.shape[0])
H_description = H_description[perm]
O_description = O_description[perm]
C_description = C_description[perm]
data_energy = data_energy[perm]

train_x_H = H_description[:number_molecules//4]
train_x_O = O_description[:number_molecules//4]
train_x_C = C_description[:number_molecules//4]
train_y_energy = data_energy[:number_molecules//4]

test_x_H = H_description[3*number_molecules//4:]
test_x_O = O_description[3*number_molecules//4:]
test_x_C = C_description[3*number_molecules//4:]
test_y_energy = data_energy[3*number_molecules//4:]


def custom_layer(tensor):
  return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    

input1 = Input(shape= (train_x_H.shape[1],), name="H_element")
l1_1 = Dense(5, activation='elu')(input1)
l1_2 = Dense(5, activation='elu')(l1_1)
l1_3 = Dense(1)(l1_2)
l1_4 = keras.layers.Lambda(custom_layer, name = 'H_atomic_energy')(l1_3) #atomic energy for H

input2 = Input(shape= (train_x_O.shape[1],), name="O_element")
l2_1 = Dense(5, activation='elu')(input2)
l2_2 = Dense(5, activation='elu')(l1_2)
l2_3 = Dense(1)(l2_2)
l2_4 = keras.layers.Lambda(custom_layer, name = 'O_atomic_energy')(l2_3) #atomic energy for O

input3 = Input(shape= (train_x_C.shape[1],), name="C_element")
l3_1 = Dense(5, activation='elu')(input3)
l3_2 = Dense(5, activation='elu')(l1_2)
l3_3 = Dense(1)(l3_2)
l3_4 = keras.layers.Lambda(custom_layer, name = 'C_atomic_energy')(l3_3) #atomic energy for C

# summation and output. Total energy is the target
x = layers.concatenate([l1_4, l2_4, l3_4])


def ReduceSum(z):
    return K.sum(z, axis=1, keepdims=True)


output = ReduceSum(x)

model = Model(inputs=[input1, input2, input3], outputs=output)

model.summary()

model.compile(loss='mape', optimizer=keras.optimizers.Adam(0.001))

start = perf_counter()

history = model.fit([train_x_H, train_x_O, train_x_C], train_y_energy, epochs=10, verbose=1, validation_split=0.1)

end = perf_counter()
print((end - start) / 60, "мин.")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

print(model.evaluate([test_x_H, test_x_O, test_x_C],test_y_energy))
