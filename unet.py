# NOTE: The image size should be selected, such that the consecutive convs and max-pooling yields even values of x and y (i.e. width and height of the feature map)at each stage. 96x96, 128x128, 256x256. Crop out the borders to get an appropriate image size

# Note: when using the categorical_crossentropy loss, your targets should be in categorical format (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample). In order to convert integer targets into categorical targets, you can use the Keras utility to_categorical
# from keras.utils import to_categorical
# categorical_labels = to_categorical(int_labels, num_classes=None)

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.utils import to_categorical
import PIL
import os



class Unet():


    def __init__(self, num_images=100, img_w=48, img_h=48, out_ch=3, start_ch=64, depth=4, inc_rate=2., activation='relu', dropout=0, batchnorm=True, maxpool=True, upconv=True, residual=False, batch_size=16, epochs=5, trainit=False):
        '''
            Credit: https://github.com/pietz/unet-keras/blob/master/unet.py

            U-Net: Convolutional Networks for Biomedical Image Segmentation
            (https://arxiv.org/abs/1505.04597)
            ---
            img_shape:  (height, width, channels)
            out_ch:     number of output channels
            start_ch:   number of channels of the first conv
            depth:      zero indexed depth of the U-structure
            inc_rate:   rate at which the conv channels will increase
            activation: activation function after convolutions
            dropout:    amount of dropout in the contracting part
            batchnorm:  adds Batch Normalization if true
            maxpool:    use strided conv instead of maxpooling if false
            upconv:     use transposed conv instead of upsamping + conv if false
            residual:   add residual connections around each conv block if true
        '''

        self.basepath = '/home/francesco/lavoriMiei/cbs/data/dataFromOthers/Training_ImDatabase_FCN_Myxo/' 
        self.num_images = num_images
        self.img_w = img_w        
        self.img_h = img_h        
        
        self.out_ch = out_ch
        self.start_ch = start_ch
        self.depth = depth
        self.inc_rate = inc_rate
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.maxpool = maxpool
        self.upconv = upconv
        self.residual = residual
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.make_sets(self.num_images)
        self.def_unet()
        if trainit:
            self.train_unet()
            self.plot_history()
            self.show_val()



    def open_one_image(self, imagefile=None, plots=False):
        ''' open, crop image'''
        if not imagefile:
            imagefile = self.imagefile
        img = PIL.Image.open(imagefile)
        # open and crop:
        img = np.array(img.getdata()).reshape(img.size)
        img = img[:self.img_h, :self.img_w]
        if plots:
            plt.figure('open_one_image', clear=True)
            plt.imshow(img)
            plt.colorbar()
        return img


    
    def make_sets(self, num_images=100):
        ''' read, crop, and store training testing npy images 
            num_images : take all if None
            TODO include only files with 1,120,220 without converting labels
        '''
        l_train_orig = np.sort(os.listdir(self.basepath+'Training/Original/'))
        if num_images == None:
            num_images = len(l_train_orig)
        # init sets:
        self.train_set = np.zeros((num_images, self.img_h, self.img_w))
        self.train_lab = np.zeros((num_images, self.img_h, self.img_w))
        # open store images:
        for i in range(num_images):
            f = l_train_orig[i]
            print(f'make_sets(): Loading {f} {i}/{num_images-1}', end='\r')
            lab = self.open_one_image(self.basepath + 'Training/Labeled/' + f)
            # convert 100 200 labels (e.coli):
            lab[np.nonzero(lab == 100)] = 120            
            lab[np.nonzero(lab == 200)] = 220            
            self.train_set[i] = self.open_one_image(self.basepath + 'Training/Original/' + f)
            self.train_lab[i] = lab
        # normalize training set:
        self.train_set = (self.train_set - np.mean(self.train_set, axis=0))/np.std(self.train_set, axis=0)
        self.train_set = self.train_set[:,:,:,np.newaxis]
        # normalize training labels:
        self.num_labels = len(np.unique(self.train_lab))
        for i, l in enumerate(np.unique(self.train_lab)):
            self.train_lab[np.nonzero(self.train_lab == l)] = i
        # labels to categorical:
        self.train_lab = to_categorical(self.train_lab, self.num_labels)
        print(f'\nmake_sets(): found {self.num_labels} labels')
         



    def def_unet(self):

        def conv_block(m, dim, acti, bn, res, do=0):
            n = Conv2D(dim, 3, activation=acti, padding='same')(m)
            n = BatchNormalization()(n) if bn else n
            n = Dropout(do)(n) if do else n
            n = Conv2D(dim, 3, activation=acti, padding='same')(n)
            n = BatchNormalization()(n) if bn else n
            return Concatenate()([m, n]) if res else n
        
        def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
            if depth > 0:
                n = conv_block(m, dim, acti, bn, res)
                m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
                m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
                if up:
                    m = UpSampling2D()(m)
                    m = Conv2D(dim, 2, activation=acti, padding='same')(m)
                else:
                    m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
                n = Concatenate()([n, m])
                m = conv_block(n, dim, acti, bn, res)
            else:
                m = conv_block(m, dim, acti, bn, res, do)
            return m

        print('def_unet(): model init...')
        img_shape = (self.img_w, self.img_h, 1)
        out_ch = self.out_ch
        start_ch = self.start_ch
        depth = self.depth
        inc_rate = self.inc_rate
        activation = self.activation
        dropout = self.dropout
        batchnorm = self.batchnorm
        maxpool = self.maxpool
        upconv = self.upconv
        residual = self.residual

        i = Input(shape=img_shape)
        o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
        o = Conv2D(out_ch, 1, activation='softmax')(o)
        # model:
        self.model = Model(inputs=i, outputs=o)
        # compile:
        print('def_unet(): model compile...')
        self.model.compile(optimizer='Adamax', loss='categorical_crossentropy')
        print('def_unet(): done.')
    
    

    def train_unet(self):
        self.model.fit( x=self.train_set, 
                        y=self.train_lab, 
                        batch_size=self.batch_size, 
                        epochs=self.epochs, 
                        verbose=1, 
                        validation_split=0.2 )



    def plot_history(self):
        loss = self.model.history.history['loss']
        val_loss = self.model.history.history['val_loss']
        epochs = self.model.history.epoch
        plt.figure('plot_history', clear=False)
        plt.semilogy(epochs, loss, label='loss')
        plt.semilogy(epochs, val_loss, label='val_loss')
        plt.xlabel('epochs')
        plt.legend()

   

    def show_val(self, imgidx=0):
        img  = self.model.history.validation_data[0][imgidx,...,0]
        lab1 = self.model.history.validation_data[1][imgidx,...,0]
        lab2 = self.model.history.validation_data[1][imgidx,...,1]
        lab3 = self.model.history.validation_data[1][imgidx,...,2]
        lab4 = self.model.history.validation_data[1][imgidx,...,3]
        loss = self.model.history.history['loss']
        val_loss = self.model.history.history['val_loss']
        epochs = self.model.history.epoch
        fig = plt.figure('show_val', clear=True, figsize=(7.5,6))
        ax1 = fig.add_subplot(321)
        ax1.imshow(img)
        ax1.set_title(f'val[{imgidx}]')
        ax2 = fig.add_subplot(322)
        ax2.imshow(img + 4*lab4)
        ax2.set_title('img+lab4')
        ax3 = fig.add_subplot(345)
        ax3.imshow(lab1)
        ax3.set_title('lab1')
        ax4 = fig.add_subplot(346)
        ax4.imshow(lab2)
        ax4.set_title('lab2')
        ax5 = fig.add_subplot(347)
        ax5.imshow(lab3)
        ax5.set_title('lab3')
        ax6 = fig.add_subplot(348)
        ax6.imshow(lab4)
        ax6.set_title('lab4')
        ax7 = fig.add_subplot(313)
        ax7.semilogy(epochs, loss, label='loss')
        ax7.semilogy(epochs, val_loss, label='val_loss')
        ax7.set_xlabel('epochs')
        ax7.legend()


    def check_predition(self, imgidx=0):
        pred = self.model.predict()

