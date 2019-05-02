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

    def __init__(self, model_type='dense', epochs=10):
        self.epochs = epochs
        self.load_data()
        self.model_type = model_type
        if model_type == 'dense':
            self.make_dense_model()
        elif model_type == 'conv':
            self.make_conv_model()
        else:
            raise Exception('model_type not valid')
        print(self.model.summary())
        self.compile_train_plot_model()



    def load_data(self):
        '''download the data'''
        from keras.datasets import mnist
        print('Downloading data...')
        (self.train_imgs, self.train_labels), (self.test_imgs, self.test_labels) = mnist.load_data()
        print('Done.')



    def make_dense_model(self):
        ''' densely connected model '''
        print('Init. densely connected model...')
        # prepare data:
        self.train_imgs = self.train_imgs.reshape((60000, 28*28))
        self.train_imgs = self.train_imgs.astype('float32')/255
        self.test_imgs = self.test_imgs.reshape((10000, 28*28))
        self.test_imgs = self.test_imgs.astype('float32')/255
        self.train_labels = keras.utils.to_categorical(self.train_labels)
        self.test_labels  = keras.utils.to_categorical(self.test_labels)
        # build model:
        self.model = keras.models.Sequential() 
        self.model.add(keras.layers.Dense(512, activation='relu', input_shape=(28*28,)))   
        # add dropout:
        #self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        print('Done.')



    def make_conv_model(self):
        ''' convolutional + densely connected model '''
        print('Init. convolutional + densely connected model...')
        # prepare data:
        self.train_imgs = self.train_imgs.reshape((60000, 28, 28, 1))
        self.train_imgs = self.train_imgs.astype('float32')/255
        self.test_imgs = self.test_imgs.reshape((10000, 28, 28, 1))
        self.test_imgs = self.test_imgs.astype('float32')/255
        self.train_labels = keras.utils.to_categorical(self.train_labels)
        self.test_labels  = keras.utils.to_categorical(self.test_labels)
        # build model:
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(512, activation='relu'))
        #self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        print('Done.')



    def compile_train_plot_model(self):
        # compile the model:
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # train the model: 
        self.model.fit(self.train_imgs, self.train_labels, validation_split=0.1, epochs=self.epochs, batch_size=128)
        # plot losses:
        plt.figure('make_model')
        plt.subplot(211)
        plt.plot(self.model.history.history['loss'], label=self.model_type+' train_loss')
        plt.plot(self.model.history.history['val_loss'], label=self.model_type+' val_loss')
        plt.legend()
        plt.subplot(212)
        # metric on training set:
        plt.plot(self.model.history.history['acc'], label=self.model_type+' train_acc')
        # metric on validation set:
        plt.plot(self.model.history.history['val_acc'], label=self.model_type+' val_acc')
        plt.legend()
        plt.subplot(212)
        plt.xlabel('epoch')



    def predict(self, n=0):
        ''' show test image number n with its prediction'''
        plt.imshow(self.test_imgs[n].reshape(28,28)) 
        plt.title(f'True: {np.where(self.test_labels[n])[0]}   Predicted:{np.argmax(self.model.predict(self.test_imgs[n:n+1]))}')
