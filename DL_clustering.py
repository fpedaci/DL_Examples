import keras
import numpy as np
import matplotlib.pyplot as plt


class DL_clustering():
    '''
    Deep learning clustering of simulated data
    
    example usage:
    
    import DL_clustering

    cl = DL_clustering.DL_clustering(pts_cluster=1000, epochs=15, cl_width=[2,2,2,3,2,2,2,3,2,3,2], cl_pos = [(0,0),(0,20),(20,-5),(15,25),(20,10),(40,0),(-20,0),(-20,15),(-10,28),(31,20),(40,12)])

    cl = DL_clustering.DL_clustering(pts_cluster=100, cl_width=[3,3,3,3,2,3,2,2,3,2,3,3,2,3,2,3,3,3,3,3,3,3,3,3,3,3,3,3,1], cl_pos=[(-30,-30),(0,8),(0,20),(20,-5),(15,25),(20,10),(40,0),(-20,0),(-20,15),(-10,28),(36,25),(40,12),(-10,-10),(8,-13),(30,-15),(0,36),(25,35),(-21,-21),(3,-26),(22,-28),(-22,35),(43,-28),(-10,-30),(45,35),(-32,10),(-30,-10),(-31,25),(44,-15),(-3,-3)], epochs=15)
     '''
    
    def __init__(self, pts_cluster=500, cl_width=[1,], cl_pos=[(0,0),], epochs=10, savefiles=False, verbose=False):
        ''' cl_pos = list [(x1,y1), ...] of positions of each cluster. controls the number of clusters
            cl_width = number or list (must be of len(cl_pos) ), widths of each clusters
            pts_cluster : points in each cluster
            epochs : n. of epochs for learning
            savefiles = save or not images
        '''    
        if len(cl_pos) > len(cl_width):
            print('warning! cl_pos longer than cl_width')
            cl_pos = cl_pos[:len(cl_width)]
        if len(cl_pos) < len(cl_width):
            print('warning! cl_width longer cl_pos') 
            cl_width = cl_width[:len(cl_pos)]
        if type(cl_width) == float or type(cl_width) == int:
            cl_width = np.repeat(cl_width, len(cl_pos))

        self.n_clusters = len(cl_pos)
        self.pts_cluster = pts_cluster
        self.cl_width = cl_width
        self.cl_pos = cl_pos
        self.epochs = epochs
        self.val_loss_li = []
        self.loss_li = []
        self.epochs_li = []
        self.savefiles = savefiles
        self.verbose = verbose

        self.make_clusters()
        self.plot_clusters()
        self.norm_clusters()
        self.make_grid()
        self.make_sets()
        self.make_train_model(plots=1)
        self.unnorm_clusters() 
        self.predict(-1,-1)


    def make_grid(self):
        ''' make a (gn*gn) grid for contour plot'''
        if self.verbose: print('make_grid')
        self.gn = 300    
        self.gx = np.linspace(self.xs_min, self.xs_max, self.gn)
        self.gy = np.linspace(self.ys_min, self.ys_max, self.gn)
        xx, yy = np.meshgrid(self.gx, self.gy)
        xx = (xx - self.xs_mn)/self.xs_std
        yy = (yy - self.ys_mn)/self.ys_std 
        self.grid = np.array([i for i in zip(xx.flatten(), yy.flatten())])


    def make_clusters(self, plots=False):
        ''' random gaussian clusters defined by cl_width, cl_pos'''
        if self.verbose: print('make_clusters')
        self.clusters = {}
        for c in range(self.n_clusters):
            Xs, Ys = np.random.randn(2, self.pts_cluster)*self.cl_width[c]
            Xs = Xs + self.cl_pos[c][0]
            Ys = Ys + self.cl_pos[c][1]
            self.clusters[c] = {}
            self.clusters[c] = [Xs, Ys]
        if plots:
            self.plot_clusters()


    def norm_clusters(self):
        '''normalize all xs, ys  '''
        if self.verbose: print('norm_clusters')
        xs = [x for c in self.clusters.keys() for x in self.clusters[c][0]]
        ys = [y for c in self.clusters.keys() for y in self.clusters[c][1]]
        self.xs_std = np.std(xs)
        self.xs_mn = np.mean(xs)
        self.xs_min = np.min(xs)
        self.xs_max = np.max(xs) 
        self.ys_std = np.std(ys)
        self.ys_mn = np.mean(ys)
        self.ys_min = np.min(ys)
        self.ys_max = np.max(ys)   
        for k in self.clusters.keys():
            self.clusters[k][0] = (self.clusters[k][0] - self.xs_mn)/self.xs_std
            self.clusters[k][1] = (self.clusters[k][1] - self.ys_mn)/self.ys_std
 
    
    def unnorm_clusters(self):
        '''opposite of normalization '''
        if self.verbose: print('unnorm_clusters')
        for k in self.clusters.keys():
            self.clusters[k][0] = self.clusters[k][0] * self.xs_std + self.xs_mn
            self.clusters[k][1] = self.clusters[k][1] * self.ys_std + self.ys_mn


    def plot_clusters(self):
        if self.verbose: print('plot_clusters')
        self.fig = plt.figure('DLclustering', clear=True, figsize=(5,6.5))
        self.ax = self.fig.add_axes([.08,.31,.9,.68])
        for c in self.clusters.keys():
            self.ax.plot(self.clusters[c][0], self.clusters[c][1], '.', mew=0, alpha=0.6)
        self.ax2 = self.fig.add_axes([.1,.07,.85,.18])
        self.ax2.set_xlabel('Epoch', fontsize=8)
   

    def make_sets(self):
        ''' make train test sets'''
        if self.verbose: print('make_sets')
        for c in self.clusters.values():
            labels_x_y = np.array([(c, self.clusters[c][0][i], self.clusters[c][1][i]) for i in range(self.pts_cluster) for c in self.clusters])
            self.train_set = labels_x_y[:,1:]
            self.train_labels = keras.utils.to_categorical(labels_x_y[:,0])


    def make_train_model(self, plots=False):
        if self.verbose: print('make_train_model')
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(64, input_shape=(2,), activation='relu'))
        self.model.add(keras.layers.Dense(8, activation='relu'))
        self.model.add(keras.layers.Dense(self.n_clusters, activation='softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        # train the model: 
        callback_pred = keras.callbacks.LambdaCallback(on_epoch_end=self.predict)
        self.model.fit(self.train_set, self.train_labels, epochs=self.epochs, batch_size=32, validation_split=0.15, callbacks=[callback_pred])


    def predict(self, epochs, logs):
        ''' predict xs,xy on a grid '''
        if self.verbose: print('predict')
        # predictions on grid:
        self.pred = np.argmax(self.model.predict(self.grid), axis=1)
        self.pred = np.reshape(self.pred, (self.gn, self.gn)) 
        # plot contour, only last one:
        if hasattr(self, '_contour'):
            for coll in self._contour.collections:
                coll.remove()
        self._contour = self.ax.contour(self.gx, self.gy, self.pred, levels=self.n_clusters, colors='g')
        # plot loss:
        if epochs != -1:
            self.epochs_li.append(epochs)
            self.val_loss_li.append(logs['val_loss'])
            self.loss_li.append(logs['loss'])
            self.ax2.cla()
            self.ax2.semilogy(self.epochs_li, self.loss_li, '-', label='loss')
            self.ax2.semilogy(self.epochs_li, self.val_loss_li, '--', label='val_loss')
            self.ax2.legend(fontsize=8)
        if self.savefiles:
            if   epochs < 10: filename = f'DLclustering_00{epochs}.png'
            elif epochs < 100: filename = f'DLclustering_0{epochs}.png'
            plt.savefig(filename)
        plt.pause(0.01)


