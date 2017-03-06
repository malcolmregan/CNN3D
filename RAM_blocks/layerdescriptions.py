import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import matplotlib.pyplot as plt
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

#with open('./bmnist20161220-114356/bmnist', "rb") as f:
with open('./LeNet20161221-121708/LeNet', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

probs = lenet.apply(x)
ff = theano.function([x],[probs])

act=lenet.layers[0].apply(x)
for i in range(1,len(lenet.layers)):
    act = lenet.layers[i].apply(act)
pp = theano.function([x],[act])



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


aa=ff(xx)
print(aa[0][0])

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')
act = lenet.layers[0].apply(x)
for i in range(0,len(lenet.layers)):
    act = lenet.layers[i].apply(act)

pp=theano.function([x],[act])

aa=pp(xx)
print(np.shape(aa))

W4.set_value(W4a*0)
#W3.set_value(W3a*0)
#W2.set_value(W2a*0)
#W1.set_value(W1a*0)
b4.set_value(b4a*0)
#b3.set_value(b3a*0)
#b2.set_value(b2a*0)
#b1.set_value(b1a*0)

probs = lenet.apply(x)
ff = theano.function([x],[probs])
aa=ff(xx)
print(aa[0][0])


"""
for i in range(0, len(lenet._children)):
    if (lenet._children[i]._children):
        print(lenet._children[i].name)
        for k in range(0,len(lenet._children[i]._children._items)):
            print("\t"+str(lenet._children[i]._children._items[k].name))
            for q in lenet._children[i]._children._items[k].__dict__.iteritems():
                print("\t\t"+str(q))
            if len(lenet._children[i]._children._items[k]._parameters)>0:
                W=np.asarray(lenet._children[i]._children._items[k]._parameters[0].eval()) 
                b=np.asarray(lenet._children[i]._children._items[k]._parameters[1].eval()) 
                #lenet._children[i]._children._items[k]._parameters[0].set_value(W*72193817.0)
                print("\t\t\tW size: "+str(np.shape(W)))
                print("\t\t\tb size: "+str(np.shape(b)))
                act = lenet._children[i]._children._items[k].apply(act)
                ff = theano.function([x],[act])
                aa=ff(xx)
                print(aa[0][0][0])
"""
