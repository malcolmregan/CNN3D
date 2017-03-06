import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import matplotlib.pyplot as plt
fig=plt.figure()
import numpy as np
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, PARAMETER, FILTER
from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from matplotlib import gridspec
import math
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

probs = lenet.apply(x)
cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
        .copy(name='cost'))
error_rate = (MisclassificationRate().apply(y.flatten(), probs)
              .copy(name='error_rate'))
cg = ComputationGraph([cost, error_rate])

param= VariableFilter(roles=[PARAMETER])(cg.variables)

b4=param[0]
b3=param[2]
b2=param[4]
b1=param[6]
W4=param[1]
W3=param[3]
W2=param[5]
W1=param[7]

b4a=np.asarray(b4.eval())
b3a=np.asarray(b3.eval())
b2a=np.asarray(b2.eval())
b1a=np.asarray(b1.eval())
W4a=np.asarray(W4.eval())
W3a=np.asarray(W3.eval())
W2a=np.asarray(W2.eval())
W1a=np.asarray(W1.eval())

test_set = H5PYDataset('LeNet_CLASS8.hdf5', which_sets=('test',))
handle = test_set.open()

test_set2 = H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
handle2 = test_set2.open()

for i in range(1,len(lenet.layers)):  
    act = lenet.layers[i].apply(act)
    ff = theano.function([x],[act])
act = lenet._children[1].apply(act)
act = lenet._children[2]._children[0].apply(act)
act = lenet._children[2]._children[1].apply(act)
ff = theano.function([x],[act])

CC=np.zeros((1,500))
#average over bmnist examples
count=0
model_idx=0
while count<test_set.num_examples:
    test_data2 = test_set2.get_data(handle2, slice(model_idx, model_idx+1))
    if test_data2[1][0][0]==8:
        XX = test_data2[0]
        YY = test_data2[1][0][0]
        RR=ff(XX)[0]
        CC=CC+RR
        count=count+1
    model_idx=model_idx+1
CC=CC/count

print test_set.num_examples
print count

cc=np.zeros((1,500))
#average over cppn examples
for model_idx in range(test_set.num_examples):
    test_data = test_set.get_data(handle, slice(model_idx , model_idx +1))
    xx = test_data[0]
    yy = test_data[1][0][0]
    rr=ff(xx)[0]
    cc=cc+rr
cc=cc/test_set.num_examples




cc = np.vstack((cc,cc,cc,cc,cc))
cc = np.vstack((cc,cc,cc,cc,cc))
cc = np.vstack((cc,cc,cc))
cc = np.vstack((cc,cc))


CC = np.vstack((CC,CC,CC,CC,CC))
CC = np.vstack((CC,CC,CC,CC,CC))
CC = np.vstack((CC,CC,CC))
CC = np.vstack((CC,CC))

print np.shape(cc)
print np.shape(cc)

ax=[0]*2
#CPPN last layer before output (bottom)
ax[0] = plt.subplot2grid((2,1),(0,0))
ax[0].imshow(cc, cmap='Greys', interpolation='nearest')
plt.xticks([],[])
plt.yticks([],[])

#BMNIST last layer before output (top)
ax[1] = plt.subplot2grid((2,1),(1,0))
ax[1].imshow(CC, cmap='Greys', interpolation='nearest')
plt.xticks([],[])
plt.yticks([],[])

plt.subplots_adjust(left=0, bottom=None, right=1, top=None)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('Figure_2.png')
plt.close()

#plt.show()
