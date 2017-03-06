import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from fuel.datasets.hdf5 import h5py
from LeNet_Mnist import LeNet
from neat import nn, population, statistics
import os
import numpy as np
import math
import RAM_model as RAM_model

np.set_printoptions(threshold=np.nan)

with open('./bmnist20161223-103232/bmnist', "rb") as f:
#with open('./mnist20161223-104402/mnist', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

ram = model.get_top_bricks()[0]
x = tensor.tensor4('input')   # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
l, prob, rho_orig, rho_larger, rho_largest = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
f = theano.function([x], [l, prob, rho_orig, rho_larger, rho_largest])

#CPPN input
inp=[0]*(28*28)
n=0
for i in range(0, 28):
    for j in range(0, 28):
            inp[n]=(i,j)
            n=n+1

def get_fitness(g, inp, CLASS):
    net = nn.create_feed_forward_phenotype(g)
    outputarray = [0]*28*28
    for inputs in inp:
        output = net.serial_activate(inputs) 
        outputarray[inputs[0]+inputs[1]*28] = output[0]
    outputarray = np.reshape(outputarray,(28,28))
    threshold=0.5
    outputarray[outputarray<threshold]=0
    outputarray[outputarray>=threshold]=1
    feat=np.zeros((1,1,28,28),dtype=np.uint8)
    feat[0,0,:,:]=outputarray
    l, prob, rho_orig, rho_larger, rho_largest = f(feat)
    pred=prob[9][0]
    pred[:CLASS]=np.absolute(pred[:CLASS])
    pred[(CLASS+1):]=np.absolute(pred[(CLASS+1):])
    fitness = (pred[CLASS]-min(pred))/(np.sum(pred-min(pred)))
    save_path=str(os.path.dirname(os.path.realpath(__file__)))+'/CPPNGenerated/RAM/class{0}'.format(CLASS)
    file_count = len(os.walk(save_path).next()[2])
    if fitness>.987:
        filename = "class{0}_{1}".format(CLASS, file_count)
        temp = np.zeros((1,1,28,28),dtype=np.uint8)
        temp[0,0,:,:] = outputarray
        np.savez(os.path.join(save_path,filename+'.npz'),**{'features': temp, 'targets': [CLASS]})
        fitness=100
        global flag
        flag=1
    if pop.generation == 4000:
        fitness=100
        global flag
        flag=0
    return fitness

clazz=6

def eval_fitness(genomes):
    for g in genomes:
        g.fitness = get_fitness(g, inp, clazz)

while 1:
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'RAM_CPPN_config')
    pop = population.Population(config_path)
    pop.run(eval_fitness, 4001)   
    if flag==1:
        clazz=(clazz+1)%10
