import keras
import matplotlib.pyplot as plt
import numpy as np

class mnist():
    ''' simple sequential model to learn mnist dataset 
    usage:
        import mnist
        mn = mnist.mnist()
        mn.predict(3)
        mn.predict(10) 
        ...
    '''

    def __init__(self):
        self.load_data()
        self.make_model()
        print('\nDone. Now test some image \'with mn.predict(index)\' ')


    def load_data(self):
        '''download the data'''
        from keras.datasets import mnist
        print('Downloading data...')
        (self.train_imgs, self.train_labels), (self.test_imgs, self.test_labels) = mnist.load_data()


    def make_model(self):
        # prepare data:
        self.train_imgs = self.train_imgs.reshape((60000, 28*28))
        self.train_imgs = self.train_imgs.astype('float32')/255
        self.test_imgs = self.test_imgs.reshape((10000, 28*28))
        self.test_imgs = self.test_imgs.astype('float32')/255
        self.train_labels = keras.utils.to_categorical(self.train_labels)
        self.test_labels  = keras.utils.to_categorical(self.test_labels)

        # build model:
        print('Init model...')
        self.model = keras.models.Sequential() 
        self.model.add(keras.layers.Dense(512, activation='relu', input_shape=(28*28,)))   
        self.model.add(keras.layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # train the model: 
        self.model.fit(self.train_imgs, self.train_labels, validation_split=0.1, epochs=10, batch_size=128)
        # plot losses:
        plt.figure('make_model')
        plt.subplot(211)
        plt.plot(self.model.history.history['loss'], label='loss')
        plt.plot(self.model.history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.subplot(212)
        plt.plot(self.model.history.history['acc'], label='accuracy')
        plt.legend()
        plt.subplot(212)
        plt.xlabel('epoch')


    def predict(self, n=0):
        '''show test image number n with its prediction'''
        plt.imshow(self.test_imgs[n].reshape(28,28)) 
        plt.title(f'True: {np.where(self.test_labels[n])[0]}   Predicted:{np.argmax(self.model.predict(self.test_imgs[n:n+1]))}')
