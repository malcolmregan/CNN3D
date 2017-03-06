import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import math

fig=plt.figure()

CLASS=9

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
qq = theano.function([x], [probs])

for i in range(1,6):  
    act = lenet.layers[i].apply(act)
    
ff = theano.function([x],[act])

test_set = H5PYDataset('LeNet_CLASS{0}.hdf5'.format(CLASS), which_sets=('test',))
handle = test_set.open()

num=test_set.num_examples
ax_act=[0]*((num*2)+3)

for model_idx in range(1,num+1):

    test_data = test_set.get_data(handle, slice(model_idx-1, model_idx))
    xx = test_data[0]
    YY = test_data[1][0][0]

    aa = ff(xx)[0][0][0]
    aa = aa.reshape(1,100)
    aa = np.vstack((aa,aa,aa,aa,aa)) #bad way of sizing plots     

    pp=qq(xx)[0][0]

    info='Dataset: LeNet CPPN\nClassification Confidence: {0:.2f}%'.format(pp[YY]*100)

    rows=int(math.ceil(float(num)/2))
    
    if model_idx>rows:
        colu=2
    else:
         colu=0

    #ax_act[model_idx*2-1] = fig.add_subplot(8,4,(model_idx*2)+3)
    ax_act[model_idx*2-1] = plt.subplot2grid((rows+1,4),(((model_idx-1)%rows)+1,colu), colspan=1)
    ax_act[model_idx*2-1].imshow(xx[0][0], cmap='Greys', interpolation='nearest')    
    plt.xticks([],[])
    plt.yticks([],[])
    
    #ax_act[model_idx*2] = fig.add_subplot(8,4,(model_idx*2)+4)
    ax_act[model_idx*2] = plt.subplot2grid((rows+1,4),(((model_idx-1)%rows)+1,colu+1), colspan=1, rowspan=1)
    ax_act[model_idx*2].imshow(aa, cmap='Greys', interpolation='nearest')
    plt.xticks([],[])
    plt.yticks([],[])
    
    ax_act[model_idx*2-1].set_xlabel(info)

test_set = H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
handle = test_set.open()

for i in range(0,100):
    test_data = test_set.get_data(handle, slice(i, i+1))
    if test_data[1][0][0]==CLASS:
        model_idx=i

test_data = test_set.get_data(handle, slice(model_idx,model_idx+1))    
XX = test_data[0]
YY = test_data[1][0][0]

AA = ff(XX)[0][0][0]
AA = AA.reshape(1,100)
AA = np.vstack((AA,AA,AA,AA,AA))


pp = qq(XX)[0][0]

info='Dataset: BMNIST\nClassification Confidence: {0:.2f}%'.format(pp[YY]*100)

#ax_act[num*2+1] = fig.add_subplot(8,4,2)
ax_act[num*2+1] = plt.subplot2grid((rows+1,4),(0,1), colspan=1)
ax_act[num*2+1].imshow(XX[0][0], cmap='Greys', interpolation='nearest')
plt.xticks([],[])
plt.yticks([],[])

#ax_act[num*2+2] = fig.add_subplot(8,4,3)
ax_act[num*2+2] = plt.subplot2grid((rows+1,4),(0,2), colspan=1)
ax_act[num*2+2].imshow(AA, cmap='Greys', interpolation='nearest')
plt.xticks([],[])
plt.yticks([],[])

ax_act[num*2+1].set_xlabel(info)

#plt.suptitle('5th Layer Activation for LeNet (Class {0})'.format(CLASS))
#fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.05, hspace=.7)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.suptitle('LeNet, 5th Layer Activation, Class {0}'.format(CLASS, model_idx), size=25)
plt.savefig('images/LeNetlayers/5thlayer/LeNet_Class{0}.png'.format(CLASS))
plt.close()

plt.show()
