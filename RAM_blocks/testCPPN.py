import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import RAM_model as RAM_model
import numpy as np

CPPN='LeNet'
#CPPN='RAM'
#CPPN='Draw'

#MODEL='LeNet'
MODEL='RAM'
#MODEL='Draw'

if MODEL=='LeNet':
    with open('./LeNet20161221-121708/LeNet', "rb") as f:
        p = load(f, 'model')
    if isinstance(p, Model):
        model = p

    lenet = model.get_top_bricks()[0]

    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    probs = lenet.apply(x)
    f = theano.function([x], [probs])

elif MODEL=='RAM':
    with open('./bmnist20161223-103232/bmnist', "rb") as f:
        p = load(f, 'model')

    if isinstance(p, Model):
        model = p

    ram = model.get_top_bricks()[0]
    x = tensor.tensor4('input')  # keyword from fuel
    y = tensor.matrix('targets')  # keyword from fuel
    l, prob, rho_orig, rho_larger, rho_largest = ram.classify(x)
    f = theano.function([x], [l, prob, rho_orig, rho_larger, rho_largest])

#elif MODEL=='Draw':

for CLASS in range(0,10):
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

        if MODEL=='LeNet':
            tt = f(xx)[0]
        elif MODEL=='RAM':
            l, prob, rho_orig, rho_larger, rho_largest = f(xx)
            tt = prob[9]
        #ekif MODEL='Draw':

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
    print "Class {0} examples classified by {1} in the same manner as {2}:".format(CLASS,MODEL,CPPN),
    for i in range(0,len(CPPN_Class)):
        print "{0}, ".format(CPPN_Class[i]),
    print "({0:.4f}%)".format(CPPN_rate*100)
    print "Class {0} examples classified by {1} as random (no class):".format(CLASS,MODEL),
    for i in range(0,len(correct)):
        print "{0}, ".format(correct[i]),
    print "({0:.4f}%)".format(correct_rate*100)
    print "Class {0} examples classified by {1} as something other than {2} class:".format(CLASS,MODEL,CPPN,),
    for i in range(0,len(neither)):
        print "{0}, ".format(neither[i]),
    print "({0:.4f}%)".format(neither_rate*100)
    print " "
