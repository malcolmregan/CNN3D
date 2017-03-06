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
#CPPN='RAM'
#CPPN='Draw'
CPPN='LeNet'

pbar = ProgressBar()

# with open('./bmnist20161220-114356/bmnist', "rb") as f:
#with open('./LeNet20161221-121708/LeNet', "rb") as f: #50 epochs
#with open('./LeNet20170128-160717/LeNet', "rb") as f: #100 epochs
#with open('./LeNet20170129-160735/LeNet', "rb") as f: #20 epochs, no softmax
with open('./LeNetBMNISTplusCPPN20170211-121037/LeNetBMNISTplusCPPN', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

# Normalize input and apply the convnet
probs = lenet.apply(x)
ff = theano.function([x], [probs])

test_set=H5PYDataset('../data/bmnistplusCPPN.hdf5', which_sets=('test',))
handle=test_set.open()

ex=[0]*11
dist=[0]*11

print "Testing BMNIST plus CPPN (trained on 50 epochs)..."
count=0
for i in pbar(range(0,test_set.num_examples)):
    test_data=test_set.get_data(handle, slice(i,i+1))
    xx=test_data[0]
    YY=test_data[1][0][0]
    tt=ff(xx)[0]
    if tt[0][YY] < .95: 
        count=count+1
        dist[YY]=dist[YY]+1
    ex[YY]=ex[YY]+1

error_rate=100*float(count)/test_set.num_examples
for i in range(0,11):
    print "Class {0} Error Rate: {1}/{2} ({3}%)".format(i, dist[i], ex[i], 100*float(dist[i])/ex[i])
print "Overall Error Rate: {}%".format(error_rate)

pbar2 = ProgressBar()

with open('./LeNet20161221-121708/LeNet', "rb") as f: #50 epochs
#with open('./LeNet20170128-160717/LeNet', "rb") as f: #100 epochs
#with open('./LeNet20170129-160735/LeNet', "rb") as f: #20 epochs, no softmax
#with open('./LeNetBMNISTplusCPPN20170211-121037/LeNetBMNISTplusCPPN', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

# Normalize input and apply the convnet
probs = lenet.apply(x)
ff = theano.function([x], [probs])

test_set=H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
handle=test_set.open()

ex=[0]*10
dist=[0]*10

print "Testing BMNIST (trained on 50 epochs)..."
count=0
for i in pbar2(range(0,test_set.num_examples)):
    test_data=test_set.get_data(handle, slice(i,i+1))
    xx=test_data[0]
    YY=test_data[1][0][0]
    tt=ff(xx)[0]
    if tt[0][YY] < .95:
        count=count+1
        dist[YY]=dist[YY]+1
    ex[YY]=ex[YY]+1

error_rate=100*float(count)/test_set.num_examples
for i in range(0,10):
    print "Class {0} Error Rate: {1}/{2} ({3}%)".format(i, dist[i], ex[i], 100*float(dist[i])/ex[i])
print "Overall Error Rate: {}%".format(error_rate)

pbar3 = ProgressBar()

with open('./LeNet20170128-160717/LeNet', "rb") as f: #100 epochs
#with open('./LeNet20170129-160735/LeNet', "rb") as f: #20 epochs, no softmax
#with open('./LeNetBMNISTplusCPPN20170211-121037/LeNetBMNISTplusCPPN', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

# Normalize input and apply the convnet
probs = lenet.apply(x)
ff = theano.function([x], [probs])

test_set=H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
handle=test_set.open()

ex=[0]*10
dist=[0]*10

print "Testing BMNIST (trained on 100 epochs)..."
count=0
for i in pbar3(range(0,test_set.num_examples)):
    test_data=test_set.get_data(handle, slice(i,i+1))
    xx=test_data[0]
    YY=test_data[1][0][0]
    tt=ff(xx)[0]
    if tt[0][YY] < .95:
        count=count+1
        dist[YY]=dist[YY]+1
    ex[YY]=ex[YY]+1

error_rate=100*float(count)/test_set.num_examples
for i in range(0,10):
    print "Class {0} Error Rate: {1}/{2} ({3}%)".format(i, dist[i], ex[i], 100*float(dist[i])/ex[i])
print "Overall Error Rate: {}%".format(error_rate)


"""
for CLASS in range(0,11):
    test_set = H5PYDataset('{0}_CLASS{1}.hdf5'.format(CPPN, CLASS), which_sets=('test',))
    handle = test_set.open()

    correct=[]
    CPPN_Class=[]
    neither=[]
    model_idx=0
    num = test_set.num_examples
  
    for example in range(0,num):
        test_data = test_set.get_data(handle, slice(model_idx , model_idx +1))
        xx = test_data[0]
        YY = test_data[1][0][0]
        tt = f(xx)[0]

        if (tt[0][YY] > .9):
            CPPN_Class.append('ex{}'.format(example))
        elif np.count_nonzero(tt[0]>.9) == 0:
            correct.append('ex{}'.format(example))
        else:
            neither.append('ex{} as class{}'.format(example,np.nonzero(tt[0]>.9)[0][0]))
        model_idx = model_idx+1

    CPPN_rate = float(len(CPPN_Class))/model_idx
    correct_rate = float(len(correct))/model_idx
    neither_rate = float(len(neither))/model_idx

    print " "
    print "{0} CPPN class {1} dataset has {2} examples".format(CPPN, CLASS, num)
    print "Class {0} examples classified by LenNet in the same manner as {1}:        ".format(CLASS,CPPN),
    for i in range(0,len(CPPN_Class)):
        print "{0}, ".format(CPPN_Class[i]),
    print "({0:.4f}%)".format(CPPN_rate*100)
    print "Class {0} examples classified by LeNet as random (no class):              ".format(CLASS),
    for i in range(0,len(correct)):
        print "{0}, ".format(correct[i]),
    print "({0:.4f}%)".format(correct_rate*100)
    print "Class {0} examples classified by LeNet as something other than {1} class: ".format(CLASS,CPPN),
    for i in range(0,len(neither)):
        print "{0}, ".format(neither[i]),
    print "({0:.4f}%)".format(neither_rate*100)
    print " "
"""
