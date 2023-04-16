from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

dataset = pd.read_csv("diabetes.csv")
datalar = dataset.iloc[:, 0:10]
etiketler = dataset

dataset.head()

model = Sequential()

model.add(Dense(25, input_dim = 8, activation='relu'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dataset = model.fit(datalar, etiketler, epochs=500, batch_size=10)
basarim=model.evaluate(datalar,etiketler)

print("\n %s : %.2f%%" % (model.metrics_names[1], basarim[1]*100)) 
