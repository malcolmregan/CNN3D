import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import RAM_model as RAM_model
import numpy as np

def plotinfo(CPPN,CLASS):
    test_set = H5PYDataset('{0}_CLASS{1}.hdf5'.format(CPPN, CLASS), which_sets=('test',))
    handle = test_set.open()
    num = test_set.num_examples
    info=['']*num

    MODELS=['LeNet','RAM'] #also Draw
    for MOD in MODELS:
        print MOD
        if MOD=='LeNet':
            with open('./LeNet20161221-121708/LeNet', "rb") as f:
                p = load(f, 'model')

            if isinstance(p, Model):
                model = p

            lenet = model.get_top_bricks()[0]

            x = tensor.tensor4('features')
            y = tensor.lmatrix('targets')

            probs = lenet.apply(x)
            f = theano.function([x], [probs])

        elif MOD=='RAM':
            with open('./bmnist20161223-103232/bmnist', "rb") as f:
                p = load(f, 'model')

            if isinstance(p, Model):
                model = p

            ram = model.get_top_bricks()[0]
            x = tensor.tensor4('input')  # keyword from fuel
            y = tensor.matrix('targets')  # keyword from fuel
            l, prob, rho_orig, rho_larger, rho_largest = ram.classify(x)
            f = theano.function([x], [l, prob, rho_orig, rho_larger, rho_largest])

        #elif MOD=='Draw':
    
        model_idx=0

        for example in range(0,num):
            test_data = test_set.get_data(handle, slice(model_idx , model_idx +1))
            xx = test_data[0]
            YY = test_data[1][0][0]

            if MOD=='LeNet':
                tt = f(xx)[0]
            elif MOD=='RAM':
                l, prob, rho_orig, rho_larger, rho_largest = f(xx)
                tt = prob[9]
            #elif MOD=='Draw':

            if (tt[0][YY] > .9):
                info[example]=info[example]+'{0}: {1}   '.format(MOD, YY)
            elif np.count_nonzero(tt[0]>.9) == 0:
                info[example]=info[example]+'{0}: none  '.format(MOD)
            else:
                info[example]=info[example]+'{0}: {1}   '.format(MOD,np.nonzero(tt[0]>.9)[0][0])
            model_idx = model_idx+1

    return info
