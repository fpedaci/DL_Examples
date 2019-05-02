import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dropout


n_points = 5000
noise = 0.05

epochs = 20
activation = 'relu' # try: 'sigmoid' 
optimizer = 'adagrad'  # try: 'adam', 'sgd', 'adagrad', 'adamax'


def funct_to_fit(x):
    return (np.sin(10*x)*np.cos(4*x + 2.2))/2

# define train set:
x_train = np.random.uniform(0,1, size=n_points)
x_train = x_train.reshape(len(x_train),1)
y_train = funct_to_fit(x_train)
y_train = y_train + noise*np.random.randn(len(y_train),1)

# define validation set:
x_val = np.random.uniform(0,1, size=int(n_points/10))
x_val = x_val.reshape(len(x_val),1)
y_val = funct_to_fit(x_val)

# define test set:
x_test = np.random.uniform(0,1, size=int(n_points/10))
x_test = x_test.reshape(len(x_test),1)
y_test = funct_to_fit(x_test)

# define model:
model = keras.models.Sequential()
model.add(keras.layers.Dense(250, input_shape=(1,), activation=activation))
model.add(keras.layers.Dense(150, activation=activation))
model.add(keras.layers.Dense(80, activation=activation))
model.add(keras.layers.Dense(80, activation=activation))
model.add(keras.layers.Dense(50, activation=activation))
model.add(keras.layers.Dense(20, activation=activation))
model.add(keras.layers.Dense(20, activation=activation))
model.add(keras.layers.Dense(1))
model.compile(loss='mse', optimizer=optimizer)

# train the model:
model.fit(x_train, 
        y_train, 
        validation_split=0.1, #data=(x_val, y_val), 
        epochs=epochs, 
        batch_size=8, 
        verbose=1)

# make predictions on test set:
predictions = model.predict(x_test)

# make plots:
plt.figure('fitfunct')
plt.subplot(211)
plt.plot(x_train, y_train, 'o', label='train', ms=1, alpha=0.3)
plt.plot(x_test, predictions, '.', label='pred', alpha=0.8)
plt.legend()
plt.subplot(212)
plt.plot(model.history.history['loss'], label='loss')
plt.plot(model.history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()



