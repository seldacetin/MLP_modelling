from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(9)
dataset = np.genfromtxt('diabetes.csv', delimiter=',', skip_header=1)
data = dataset[:, 0:10]
etiketler = dataset[:,8]

model = Sequential()

model.add(Dense(25, input_dim = 9, activation='relu'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
model.fit(data, etiketler, epochs=100, batch_size=5)
basarim=model.evaluate(data, etiketler)

print("\n %s : %.2f%%" % (model.metrics_names[1], basarim[1]*100)) 
