import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import math
import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import RAM_model as RAM_model
import numpy as np

###### SET CLASS AND CPPN MODEL #######

CLASS=0

CPPN='RAM'
#CPPN='LeNet'
#CPPN='Draw'

#######################################

###### GET CLASSIFICATION INFO ########

test_set = H5PYDataset('{0}_CLASS{1}.hdf5'.format(CPPN, CLASS), which_sets=('test',))
handle = test_set.open()
num = test_set.num_examples
info=['']*num

MODELS=['LeNet','RAM'] #also Draw
for MOD in MODELS:
    print "\nTesting {0}".format(MOD)
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
            info[example]=info[example]+'{0}: {1} ({2:.1f}%)\n'.format(MOD, YY,tt[0][YY]*100)
        elif np.count_nonzero(tt[0]>.9) == 0:
            info[example]=info[example]+'{0}: none\n'.format(MOD)
        else:
            info[example]=info[example]+'{0}: {1} ({2:.1f}%)\n'.format(MOD,np.nonzero(tt[0]>.9)[0][0],tt[0][np.nonzero(tt[0]>.9)[0][0]]*100)
        model_idx = model_idx+1

################################################################

################# PLOT EXAMPLES ################################

fig = plt.figure()

datapath = 'CPPNGenerated/{0}/class{1}'.format(CPPN,CLASS)
numfiles=len(os.walk(datapath).next()[2]) 
ax=[0]*numfiles

#Change this
if numfiles<4:
    rowlength=1
elif numfiles>=4 and numfiles<7:
    rowlength=2
elif numfiles>=7 and numfiles<10:
    rowlength=3
elif numfiles>=10 and numfiles<17:
    rowlength=9
elif numfiles>=17 and numfiles<21:
    rowlength=9
elif numfiles>=21 and numfiles<25:
    rowlength=9
elif numfiles>=25 and numfiles<29:
    rowlength=9
else: 
    rowlength=9

for files in os.listdir(datapath): 
    name=str(files)
    name=name.split('.')[0]
    fileindex=int(name.split('_')[1])
    print name
    data=np.load(os.path.join(datapath, files))
    array=data['features']
    array=array[0,0,:,:]
    filled=np.nonzero(array>0)
    x=filled[1]
    y=np.absolute(np.subtract(filled[0],28))
     
    ax[fileindex] = fig.add_subplot(int(math.ceil(float(numfiles)/rowlength)),rowlength,fileindex+1, aspect='equal')
    ax[fileindex].scatter(x, y)
   
    axes = plt.gca()
    axes.set_xlim([0,28])
    axes.set_ylim([0,28])
        
    ax[fileindex].set_title(name)       
    ax[fileindex].set_xlabel(info[fileindex])    

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

fig.tight_layout()
fig.subplots_adjust(top=0.93)
plt.suptitle('{0}+CPPN, Class {1}'.format(CPPN, CLASS), size=25) 
#plt.show()
plt.savefig('images/{0}_CPPN_Class{1}.png'.format(CPPN, CLASS))
plt.close()
