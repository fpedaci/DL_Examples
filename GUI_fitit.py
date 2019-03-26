# FP 2019 made for the deep learning formation
# define a single layer model and choose by hand the 
# parameters to fit a dummy target function

#.5   .5   .3   .6   .5 
#-1  -10  -14    8   20
#.3  -.2  -.2  -0.1  -0.2


import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter 
import keras


class DLfit:
    def __init__(self, master, verbose=True):
        self.verbose = verbose
        # init frame, button, text entry:
        frame = tkinter.Frame(master)
        self.Tx = tkinter.Text(frame, height=3, width=25)
        self.Tx2 = tkinter.Text(frame, height=1, width=3)
        self.button = tkinter.Button(frame, text="Plot it!", command=self.doit)
        self.Tx.pack()
        self.Tx2.pack(side=tkinter.RIGHT)
        self.button.pack()
        # variable x:
        self.x = np.linspace(-50,50,1000)
        self.x = self.x.reshape(len(self.x),1)
        # init plot:
        self.fig = Figure(figsize=(5,6))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.ax1.set_ylabel("Y", fontsize=12)
        self.ax1.set_autoscale_on(True)
        self.ax2.set_ylabel("Y", fontsize=12)
        self.ax3.set_xlabel("X", fontsize=12)
        self.ax3.set_ylabel('error', fontsize=12)
        self.ax2.set_autoscale_on(True)
        self.ax1.set_xticklabels([])
        self.ax2.set_xticklabels([])
        # set canvas:
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        frame.pack()
        self.target_func()


    def doit(self):
        if self.verbose : print('- doit')
        self.readvals()
        self.target_func()
        self.one_layer()
        self.calc_error()
        self.canvas.draw()        
        self.fig.tight_layout()


    def target_func(self):
        '''the function to fit '''
        if self.verbose : print('target_func')
        self.func = lambda z: np.sin(z*0.1) * np.cos(z*0.05) * 0.3*np.sin(1 + z*0.06) 
        self.targetf = self.func(self.x)
        self.ax1.clear()
        self.ax1.plot(self.x, self.targetf, 'r', lw=4, label='target')
        self.ax1.set_xticklabels([])
        self.ax1.legend()


    def calc_error(self):
        ''' calculate error from fit '''
        if self.verbose : print('calc_error')
        self.error_fun = self.layer1 - self.targetf 
        self.error_val = np.sum(np.abs(self.error_fun))
        self.ax3.clear()
        self.ax3.plot(self.x, self.error_fun, '.')
        self.ax3.set_title(f'error = {self.error_val:.2f}', fontsize=10)
        self.ax3.set_ylabel("error", fontsize=12)
        self.ax3.set_xlabel("X", fontsize=12)


    def readvals(self):
        '''read the values in text entries, 
        then send to one_layer() '''
        if self.verbose : print('readvals')
        tx = self.Tx.get('1.0',tkinter.END)
        tx2 = self.Tx2.get('1.0',tkinter.END)
        self.w0 = np.array(list(map(float, tx.splitlines()[0].split())))
        self.b0 = np.array(list(map(float, tx.splitlines()[1].split())))
        self.w1 = np.array(list(map(float, tx.splitlines()[2].split())))
        self.b1 = float(tx2)


    def sigmoid(self, x):
        if self.verbose : print('sigmoid')
        return 1/(1+np.exp(-x))


    def relu(self, x):
        return (x>0)*x


    def neuron(self, w=1, b=0, activation='sigmoid'):
        '''return output of one neuron '''
        if self.verbose : print('neuron')
        if activation == 'relu':
            y = self.relu(w*self.x + b)
        elif activation == 'sigmoid':
            y = self.sigmoid(w*self.x + b)
        return y


    def one_layer(self, plots=False):    
        ''' One layer with N neurons.
        w0, b0, w1, b1 from text entry.
        plot inividual outputs of each neuron and X,Y such 

          /   activ(w0[0]+b0[0]) * w1[0]  \   
        X --  activ(w0[1]+b0[1]) * w1[1] -- (Sum + b1) = Y
          \   activ(w0[2]+b0[2]) * w1[2]  /
              ...                    ...    
        '''
        if self.verbose : print('one_layer')
        w0 = self.w0 
        b0 = self.b0 
        w1 = self.w1 
        layer1 = np.zeros(len(self.x))
        self.ax1.clear()
        self.ax2.clear()
        self.target_func()
        for i in range(len(w0)):
            layer1 = layer1 + w1[i] * self.neuron(w0[i], b0[i])
            self.ax2.plot(self.x, w1[i]*self.neuron(w0[i], b0[i]), lw=2)        
        layer1 = layer1 + self.b1
        self.layer1 = layer1
        # plots:
        self.ax1.plot(self.x, layer1, 'k--', lw=4, alpha=0.6, label='Y')
        #self.ax1.legend()
        self.ax2.plot(self.x, layer1, 'k--', lw=4, alpha=0.6)
        self.ax2.set_xticklabels([])
        self.ax2.set_ylabel("Y", fontsize=12)
        self.canvas.draw()        


    def deep(self):
        '''using DL to fit self.func() ''' 
        if self.verbose : print('deep')
        # def sets:
        x_train = self.x[::10]
        y_train = self.func(x_train)
        x_test = np.random.uniform(low=np.min(x_train), high=np.max(x_train), size=len(x_train)/10)
        y_test = self.func(x_test)
        # TODO add noise
        # define model:
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(100, input_shape=x_train.shape, activation='relu'))
        model.add(keras.layers.Dense(50, activation='relu'))
        model.add(keras.layers.Dense(1))
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x_train, y_train, epochs=40, batch_size=20, verbose=1)
        self.pred = model.predict(x_test, batch_size=1)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = DLfit(root)
    root.mainloop()


