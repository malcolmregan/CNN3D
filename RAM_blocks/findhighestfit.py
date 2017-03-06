import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import RAM_model as RAM_model
import numpy as np
from blocks.graph import ComputationGraph
from progressbar import ProgressBar
import os
import matplotlib.pyplot as plt
from math import floor

#CPPN='RAM'
#CPPN='Draw'
CPPN='LeNet'

pbar = ProgressBar()

# with open('./bmnist20161220-114356/bmnist', "rb") as f:
#with open('./LeNet20161221-121708/LeNet', "rb") as f: #50 epochs
#with open('./LeNet20170128-160717/LeNet', "rb") as f: #100 epochs
#with open('./LeNet20170129-160735/LeNet', "rb") as f: #20 epochs, no softmax
#with open('./LeNetBMNISTplusCPPN20170211-121037/LeNetBMNISTplusCPPN', "rb") as f:
with open('./LeNetBMNISTplusCPPN220170228-121218/LeNetBMNISTplusCPPN2', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

# Normalize input and apply the convnet
probs = lenet.apply(x)
ff = theano.function([x], [probs])

fig=plt.figure()
ax=[0]*10
for CLASS in range(0,10):
    path=os.path.join('./CPPNGenerated/LeNet/moreefficient2', 'class{}'.format(CLASS))
    best=0
    print "CLASS {}".format(CLASS)
    for files in os.listdir(path):
        if files.endswith(".npz"):
            example=np.load(os.path.join(path,files))
            example=example['features'][0]
            temp = np.zeros((1,1,28,28),dtype=np.uint8)
            temp[0,0,:,:] = example[0,:,:]
            prob=ff(temp)[0][0][CLASS]
            if prob>best:
                best=prob
                bestexample=files
                print best, bestexample  
    if best!=0:
        data=os.path.join(path,bestexample)
        string=str(data)
        data=np.load(data)
        array=data['features']
        array=array[0,0,:,:]
        row=int(floor((CLASS)/5))
        col=(CLASS)%5
        ax[CLASS] = plt.subplot2grid((2,5),(row,col), colspan=1)
        ax[CLASS].imshow(array, cmap='Greys', interpolation='nearest')
        plt.xticks([],[])
        plt.yticks([],[])
        ax[CLASS].set_xlabel("CLASS {}\n{}".format(CLASS,best))
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.suptitle('CPPN Generated Examples for LeNet+CPPNx2',size=25)
#plt.show()
plt.savefig('LeNetCPPN2')

