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
func=theano.function([x],[probs])
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
#test_set = H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
handle = test_set.open()
model_idx = 6
test_data = test_set.get_data(handle, slice(model_idx , model_idx +1))
xx = test_data[0]
YY = test_data[1][0][0]

for i in range(1,len(lenet.layers)):  
    act = lenet.layers[i].apply(act)
    ff = theano.function([x],[act])
aa = ff(xx)

ax=[0]*150
#CPPN last convolutional layer feature maps (bottom)
for i in range(0,50):
    ax[i] = plt.subplot2grid((15,10),(8+int(math.floor(i/10)),i%10), colspan=1, rowspan=1)
    ax[i].imshow(aa[0][0][i], cmap='Greys', interpolation='nearest')
    plt.xticks([],[])
    plt.yticks([],[])

#print np.shape(W1a)
#print np.shape(W2a)
#print np.shape(W3a)
#print np.shape(b3a)
#print np.shape(W4a)

test_set2 = H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
handle = test_set2.open()

for i in range(0,100):
    test_data2 = test_set2.get_data(handle, slice(i, i+1))
    if test_data2[1][0][0]==8:
        model_idx=i

test_data2 = test_set2.get_data(handle, slice(model_idx,model_idx+1))
XX = test_data2[0]
YY = test_data2[1][0][0]

print("Original BMNIST example confidence: {}".format(func(XX)[0][0][YY]))
print("Original CPPN example confidence: {}".format(func(xx)[0][0][YY]))


AA = ff(XX)
#BMNIST last convolutional layer feature maps (top)
for i in range(0,50):
    ax[i+50] = plt.subplot2grid((15,10),(0+int(math.floor(i/10)),i%10), colspan=1, rowspan=1)
    ax[i+50].imshow(AA[0][0][i], cmap='Greys', interpolation='nearest')
    plt.xticks([],[])
    plt.yticks([],[])

act = lenet._children[1].apply(act)
act = lenet._children[2]._children[0].apply(act)
act = lenet._children[2]._children[1].apply(act)
ff = theano.function([x],[act])
cc = ff(xx)[0]
CC = ff(XX)[0]

cc = np.vstack((cc,cc,cc,cc,cc))
cc = np.vstack((cc,cc,cc,cc,cc))

CC = np.vstack((CC,CC,CC,CC,CC))
CC = np.vstack((CC,CC,CC,CC,CC))

#CPPN last layer before output (bottom)
ax[100] = plt.subplot2grid((15,10),(13,0), colspan=10, rowspan=1)
ax[100].imshow(cc, cmap='Greys', interpolation='nearest')
plt.xticks([],[])
plt.yticks([],[])

#BMNIST last layer before output (top)
ax[101] = plt.subplot2grid((15,10),(5,0), colspan=10, rowspan=1)
ax[101].imshow(CC, cmap='Greys', interpolation='nearest')
plt.xticks([],[])
plt.yticks([],[])

#modify weights and plot first fully connected layer
#print np.min(aa)
#print np.max(aa)
for i in range(0,len(aa[0][0])):
    for j in range(0,len(aa[0][0][i])):
        for k in range(0,len(aa[0][0][i][j])):
            if aa[0][0][i][j][k]<0:
                W3a[(50*i+10*j+k),:]=W3a[(50*i+10*j+k),:]*0.0
                #b3a=b3a*0.0
W3.set_value(W3a)
#b3.set_value(b3a)

act = lenet.layers[0].apply(x)
ff = theano.function([x],[act])
probs2 = lenet.apply(x)
func2 = theano.function([x],[probs2])
print("Modified weight BMNIST example confidence: {}".format(func2(XX)[0][0][YY]))
print("Modified weight CPPN example confidence: {}".format(func2(xx)[0][0][YY]))


for i in range(1,len(lenet.layers)):
    act = lenet.layers[i].apply(act)
    ff = theano.function([x],[act])

act = lenet._children[1].apply(act)
act = lenet._children[2]._children[0].apply(act)
act = lenet._children[2]._children[1].apply(act)
ff = theano.function([x],[act])

cc = ff(xx)[0]
CC = ff(XX)[0]

cc = np.vstack((cc,cc,cc,cc,cc))
cc = np.vstack((cc,cc,cc,cc,cc))

CC = np.vstack((CC,CC,CC,CC,CC))
CC = np.vstack((CC,CC,CC,CC,CC))

#CPPN last layer before output (bottom)
ax[102] = plt.subplot2grid((15,10),(14,0), colspan=10, rowspan=1)
ax[102].imshow(cc, cmap='Greys', interpolation='nearest')
plt.xticks([],[])
plt.yticks([],[])

#BMNIST last layer before output (top)
ax[103] = plt.subplot2grid((15,10),(6,0), colspan=10, rowspan=1)
ax[103].imshow(CC, cmap='Greys', interpolation='nearest')
plt.xticks([],[])
plt.yticks([],[])


plt.subplots_adjust(left=0, bottom=None, right=1, top=None)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

"""
plt.savefig('Figure_3.png')
plt.close()
"""
plt.show()

