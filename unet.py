

import numpy as np
import matplotlib.pyplot as plt
import keras
import PIL


class Unet():


    def __init__(self, imagefile=None):
        self.imagefile = imagefile
        self.basepath = '/home/francesco/lavoriMiei/cbs/data/dataFromOthers/Training_ImDatabase_FCN_Myxo/' 
        self.trainset_dir = self.basepath + 'Training'


    def open_one_image(self, imagefile=None):
        ''' '''
        if not imagefile:
            imagefile = self.imagefile
        img = PIL.Image.open(imagefile)
        img = np.array(img.getdata()).reshape(img.size)
        plt.figure('open_one_image', clear=True)
        plt.imshow(img)
        plt.colorbar()

