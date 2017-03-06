import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import matplotlib.pyplot as plt
fig=plt.figure()
import numpy as np

#with open('./bmnist20161220-114356/bmnist', "rb") as f:
with open('./LeNet20161221-121708/LeNet', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')
act = lenet.layers[0].apply(x)
ff = theano.function([x],[act])

ax_act=[0]*7
for i in range(1,6):
    print i  
    act = lenet.layers[i].apply(act)
    ff = theano.function([x],[act])

    test_set = H5PYDataset('LeNet_CLASS8.hdf5', which_sets=('test',))
    #test_set = H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
 
    handle = test_set.open()
    model_idx = 6

    test_data = test_set.get_data(handle, slice(model_idx , model_idx +1))
    xx = test_data[0]
    YY = test_data[1][0][0]

    aa = ff(xx)
    print np.shape(aa[0][0][0]) 
    ax_act[i-1] = fig.add_subplot(2,5,i+5, aspect='equal')
    ax_act[i-1].imshow(aa[0][0][0], cmap='Greys', interpolation='nearest')

ax_act[6] = fig.add_subplot(2,5,3, aspect='equal')
ax_act[6].imshow(xx[0][0], cmap='Greys', interpolation='nearest')

act=lenet.top_mlp_activations[0].apply(act)
ff=theano.function([x],[act])
aa=ff(xx)
print np.shape(aa)
print aa[0][0][0]
print " "


manager = plt.get_current_fig_manager()
manager.window.showMaximized()

fig.tight_layout()
fig.subplots_adjust(top=0.93)

#plt.show()

