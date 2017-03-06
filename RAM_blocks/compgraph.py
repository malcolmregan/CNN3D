import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
#from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from toolz.itertoolz import interleave

import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import RAM_model as RAM_model
import numpy as np
from blocks.graph import ComputationGraph

np.set_printoptions(threshold=np.nan)

with open('./LeNet20161221-121708/LeNet', "rb") as f: #50 epochs bmnist
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

# Normalize input and apply the convnet
probs = lenet.apply(x)

cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
        .copy(name='cost'))
error_rate = (MisclassificationRate().apply(y.flatten(), probs)
              .copy(name='error_rate'))
cg = ComputationGraph([cost, error_rate])

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, PARAMETER, FILTER

param= VariableFilter(roles=[PARAMETER])(cg.variables)
filt=VariableFilter(roles=[FILTER])(cg.variables)
print param

b1=param[0]
b2=param[2]
b3=param[4]
b4=param[6]
W1=param[1]
W2=param[3]
W3=param[5]
W4=param[7]

b1=np.asarray(b1.eval())
b2=np.asarray(b2.eval())
b3=np.asarray(b3.eval())
b4=np.asarray(b4.eval())
W1=np.asarray(W1.eval())
W2=np.asarray(W2.eval())
W3=np.asarray(W3.eval())
W4=np.asarray(W4.eval())

print(b1.shape)
print(b2.shape)
print(b3.shape)
print(b4.shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)
print(W4.shape)


print " "
for k in lenet.__dict__.iteritems():
    print k
print " "
print model.get_top_bricks()
print " "
print model.get_parameter_dict()
print " "
print lenet.layers[0].__dict__
print " "
print np.shape(np.asarray(lenet.layers[0]._parameters[0].eval()))
print " "
print lenet.top_mlp_activations[1].__dict__
