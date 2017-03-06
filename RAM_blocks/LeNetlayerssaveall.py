import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet
import matplotlib.pyplot as plt

#with open('./bmnist20161220-114356/bmnist', "rb") as f:
with open('./LeNet20161221-121708/LeNet', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]


for CLASS in range(0,10):
    print 'Class {0}'.format(CLASS)
 
    test_set = H5PYDataset('LeNet_CLASS{0}.hdf5'.format(CLASS), which_sets=('test',))
    #test_set = H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
    handle = test_set.open()

    for model_idx in range(0,test_set.num_examples):
        print '\tExample {0}'.format(model_idx)
        test_data = test_set.get_data(handle, slice(model_idx , model_idx +1))
        
        x = tensor.tensor4('features')
        y = tensor.lmatrix('targets')
        act = lenet.layers[0].apply(x)
        ff = theano.function([x],[act])

        fig=plt.figure()
        ax_act=[0]*7
        for i in range(1,6):
            print '\t\tLayer {0}'.format(i)
            act = lenet.layers[i].apply(act)
            ff = theano.function([x],[act])

            xx = test_data[0]
            YY = test_data[1][0][0]

            aa = ff(xx)

            ax_act[i-1] = fig.add_subplot(2,5,i+5, aspect='equal')
            ax_act[i-1].imshow(aa[0][0][0], cmap='Greys', interpolation='nearest')

        ax_act[6] = fig.add_subplot(2,5,3, aspect='equal')
        ax_act[6].imshow(xx[0][0], cmap='Greys', interpolation='nearest')

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)

        #plt.show()
        plt.suptitle('LeNet Activations, LeNet+CPPN Class {0}, Example {1}'.format(CLASS, model_idx), size=25)
        plt.savefig('images/LeNetlayers/LeNet_Class{0}_Example{1}.png'.format(CLASS,model_idx))
        plt.close()
