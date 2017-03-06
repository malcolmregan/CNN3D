import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
import RAM_model as RAM_model
from LeNet_Mnist import LeNet
import numpy

with open('./bmnist20161223-103232/bmnist', "rb") as f:
#with open('./mnist20161223-104402/mnist', "rb") as f:
        p = load(f, 'model')

if isinstance(p, Model):
    model = p

ram = model.get_top_bricks()[0]
x = tensor.tensor4('input')  # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
l, prob, rho_orig, rho_larger, rho_largest = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
f = theano.function([x], [l, prob, rho_orig, rho_larger, rho_largest])

test_set = H5PYDataset('LeNet_CLASS0.hdf5', which_sets=('test',))
handle = test_set.open()
model_idx = 0

Lenet = 0
correct = 0
neither = 0
num=test_set.num_examples
for i in range(0,num):
    test_data = test_set.get_data(handle, slice(model_idx , model_idx +1))
    xx = test_data[0]
    YY = test_data[1][0][0]
    l, prob, rho_orig, rho_larger, rho_largest = f(xx)

    if (prob[9][0][YY] > .9):
        Lenet = Lenet + 1
    elif numpy.count_nonzero(prob[9][0]>.9) == 0:
        correct=correct+1       
    else:
        neither = neither + 1
    #print prob
    model_idx = model_idx+1
Lenet_rate = float(Lenet)/model_idx
correct_rate = float(correct)/model_idx
neither_rate = float(neither)/model_idx
print " "
print "Percentage of objects classified in the same manner as LeNet: {}%".format(Lenet_rate*100)
print "Percentage of objects classified correctly as random (no class): {}%".format(correct_rate*100)
print "Percentage of objects classified as something other than LeNet class: {}%".format(neither_rate*100) 
print " "
l=l*28
################################################################################

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(5, 5)
# gs1.update(left=0.05, right=0.48, wspace=0.05)
ax_mnist = plt.subplot(gs1[0:3, 0:3])
ax_mnist.axis('equal')

ax_acc = plt.subplot(gs1[0:3,3:5])

ax_glimpse0 = plt.subplot(gs1[3,0])
ax_glimpse1 = plt.subplot(gs1[3,1])
ax_glimpse2 = plt.subplot(gs1[3,2])
ax_glimpse3 = plt.subplot(gs1[3,3])
ax_glimpse4 = plt.subplot(gs1[3,4])
ax_glimpse0.axis('equal')
ax_glimpse1.axis('equal')
ax_glimpse2.axis('equal')
ax_glimpse3.axis('equal')
ax_glimpse4.axis('equal')

ax_canvas0 = plt.subplot(gs1[4,0])
ax_canvas1 = plt.subplot(gs1[4,1])
ax_canvas2 = plt.subplot(gs1[4,2])
ax_canvas3 = plt.subplot(gs1[4,3])
ax_canvas4 = plt.subplot(gs1[4,4])
ax_canvas0.axis('equal')
ax_canvas1.axis('equal')
ax_canvas2.axis('equal')
ax_canvas3.axis('equal')
ax_canvas4.axis('equal')

ax_mnist.imshow(xx.reshape(28,28), cmap='Greys', interpolation='nearest')
# ax_mnist.set_xlim([0, 28])
# ax_mnist.set_ylim([0, 28])
for i in range(ram.n_iter-1):
    x = l[i,0,0]
    y = l[i,0,1]
    ax_mnist.text(x , y, i, fontsize=15, color='red')
    import matplotlib.patches as patches
    p = patches.Rectangle(
        (x-ram.read_N/2. , y-ram.read_N/2.), ram.read_N, ram.read_N,
        fill=False, clip_on=False, color='red'
        )
    ax_mnist.add_patch(p)

t = prob[:,0,:]
ax_acc.imshow(t.transpose(), interpolation='nearest', cmap=plt.cm.viridis,extent=[0,5,10,0])
# ax_acc.xlabel('time iteration')
# ax_acc.ylabel('class index')
# ax_acc.colorbar()

import numpy
glimpse_idx = 0
glimpse0 = numpy.zeros((28,28))
canvas0 = numpy.zeros((28,28))
x_start = l[glimpse_idx,0,1]-ram.read_N/2.
x_end = l[glimpse_idx,0,1]+ram.read_N/2.
y_start = l[glimpse_idx,0,0]-ram.read_N/2.
y_end = l[glimpse_idx,0,0]+ram.read_N/2.
glimpse_idx = glimpse_idx + 1
glimpse0[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
canvas0 = canvas0 + glimpse0
ax_glimpse0.imshow(glimpse0, cmap='Greys', interpolation='nearest')
ax_canvas0.imshow(canvas0, cmap='Greys', interpolation='nearest')
ax_glimpse0.get_xaxis().set_visible(False)
ax_glimpse0.get_yaxis().set_visible(False)
ax_canvas0.get_xaxis().set_visible(False)
ax_canvas0.get_yaxis().set_visible(False)

glimpse_idx = 1
glimpse1 = numpy.zeros((28,28))
canvas1 = numpy.zeros((28,28))
x_start = l[glimpse_idx,0,1]-ram.read_N/2.
x_end = l[glimpse_idx,0,1]+ram.read_N/2.
y_start = l[glimpse_idx,0,0]-ram.read_N/2.
y_end = l[glimpse_idx,0,0]+ram.read_N/2.
glimpse_idx = glimpse_idx + 1
glimpse1[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
canvas1 = canvas0 + glimpse1
# ax_glimpse0.get_xaxis().set_visible(False)
# ax_glimpse0.get_yaxis().set_visible(False)
ax_glimpse1.imshow(glimpse1, cmap='Greys', interpolation='nearest')
ax_canvas1.imshow(canvas1, cmap='Greys', interpolation='nearest')
ax_glimpse1.get_xaxis().set_visible(False)
ax_glimpse1.get_yaxis().set_visible(False)
ax_canvas1.get_xaxis().set_visible(False)
ax_canvas1.get_yaxis().set_visible(False)

glimpse_idx = 2
glimpse2 = numpy.zeros((28,28))
canvas2 = numpy.zeros((28,28))
x_start = l[glimpse_idx,0,1]-ram.read_N/2.
x_end = l[glimpse_idx,0,1]+ram.read_N/2.
y_start = l[glimpse_idx,0,0]-ram.read_N/2.
y_end = l[glimpse_idx,0,0]+ram.read_N/2.
glimpse_idx = glimpse_idx + 1
glimpse2[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
canvas2 = canvas1 + glimpse2
ax_glimpse2.imshow(glimpse2, cmap='Greys', interpolation='nearest')
ax_canvas2.imshow(canvas2, cmap='Greys', interpolation='nearest')
ax_glimpse2.get_xaxis().set_visible(False)
ax_glimpse2.get_yaxis().set_visible(False)
ax_canvas2.get_xaxis().set_visible(False)
ax_canvas2.get_yaxis().set_visible(False)

glimpse_idx = 3
glimpse3 = numpy.zeros((28,28))
canvas3 = numpy.zeros((28,28))
x_start = l[glimpse_idx,0,1]-ram.read_N/2.
x_end = l[glimpse_idx,0,1]+ram.read_N/2.
y_start = l[glimpse_idx,0,0]-ram.read_N/2.
y_end = l[glimpse_idx,0,0]+ram.read_N/2.
glimpse_idx = glimpse_idx + 1
glimpse3[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
canvas3 = canvas2 + glimpse3
ax_glimpse3.imshow(glimpse3, cmap='Greys', interpolation='nearest')
ax_canvas3.imshow(canvas3, cmap='Greys', interpolation='nearest')
ax_glimpse3.get_xaxis().set_visible(False)
ax_glimpse3.get_yaxis().set_visible(False)
ax_canvas3.get_xaxis().set_visible(False)
ax_canvas3.get_yaxis().set_visible(False)


# glimpse_idx = 4
# glimpse4 = numpy.zeros((28,28))
# canvas4 = numpy.zeros((28,28))
# x_start = l[glimpse_idx,0,1]-ram.read_N/2.
# x_end = l[glimpse_idx,0,1]+ram.read_N/2.
# y_start = l[glimpse_idx,0,0]-ram.read_N/2.
# y_end = l[glimpse_idx,0,0]+ram.read_N/2.
# # glimpse_idx = glimpse_idx + 1
# glimpse4[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
# canvas4 = canvas3 + glimpse4
# # ax_glimpse0.get_xaxis().set_visible(False)
# # ax_glimpse0.get_yaxis().set_visible(False)
# ax_glimpse4.imshow(glimpse4, cmap='Greys', interpolation='nearest')
# ax_canvas4.imshow(canvas4, cmap='Greys', interpolation='nearest')
# ax_glimpse4.get_xaxis().set_visible(False)
# ax_glimpse4.get_yaxis().set_visible(False)
# ax_canvas4.get_xaxis().set_visible(False)
# ax_canvas4.get_yaxis().set_visible(False)

plt.show(True)

#print(rho_orig)
