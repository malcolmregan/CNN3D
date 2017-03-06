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

MODEL='LeNet'
#MODEL='RAM'
#MODEL='Draw'

if MODEL=='LeNet':
    #with open('./LeNet20161221-121708/LeNet', "rb") as f:
    with open('./LeNetBMNISTplusCPPN20170211-121037/LeNetBMNISTplusCPPN', "rb") as f:
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
    #with open('./mnist20161223-104402/mnist', "rb") as f:
        p = load(f, 'model')

    if isinstance(p, Model):
        model = p

    ram = model.get_top_bricks()[0]
    x = tensor.tensor4('input') 
    y = tensor.matrix('targets') 
    l, prob, rho_orig, rho_larger, rho_largest = ram.classify(x)
    f = theano.function([x], [l, prob, rho_orig, rho_larger, rho_largest])

#elif MODEL=='Draw':


#CPPN input
inp=[0]*(28*28)
n=0
for i in range(0, 28):
    for j in range(0, 28):
            inp[n]=(i,j)
            n=n+1

def get_fitness(g, inp, CLASS, k):
    global gen
    if pop.generation==0:
        gen=pop.generation
    if gen!=pop.generation:
        print "Class: {}".format(CLASS)
        gen=pop.generation
    net = nn.create_feed_forward_phenotype(g)
    outputarray = [0]*28*28
    for inputs in inp:
        output = net.serial_activate(inputs)
        outputarray[inputs[0]+inputs[1]*28] = output[0]
    outputarray = np.reshape(outputarray,(28,28))
    threshold=0.5
    outputarray[outputarray<threshold]=0
    outputarray[outputarray>=threshold]=1
    feat = np.zeros((1,1,28,28),dtype=np.uint8)
    feat[0,0,:,:] = outputarray
    if MODEL=='LeNet':
        pred = f(feat)
        pred = pred[0][0] 
    elif MODEL=='RAM':
        l, prob, rho_orig, rho_larger, rho_largest = f(feat)
        pred=prob[9][0]
    #elif MODEL=='Draw':
    pred[:CLASS]=np.absolute(pred[:CLASS])     
    pred[(CLASS+1):]=np.absolute(pred[(CLASS+1):])
    fitness = (pred[CLASS]-min(pred))/(np.sum(pred-min(pred)))
    save_path=str(os.path.dirname(os.path.realpath(__file__)))+'/CPPNGenerated/{0}/moreefficient/class{1}'.format(MODEL, CLASS)
    #file_count = len(os.walk(save_path).next()[2])
    if fitness>.94: 
        filename = "{}_{}_".format(k, g.ID)
        if os.path.isfile(os.path.join(save_path,filename+'.npz'))==False:
            #Save data
            np.savez(os.path.join(save_path,filename+'.npz'),**{'features': feat, 'targets': [CLASS]})
            if fitness>.99:
                global flag
                flag=1
                fitness=100   
    if pop.generation == 50000:
        fitness=100
        global flag
        flag=1 #change back to 10 - i just didn't want it to get stuck in a loop on class 9 all night
    return fitness

clazz=5
k=22

def eval_fitness(genomes):
    for g in genomes:
        i=0
        g.fitness = get_fitness(g, inp, clazz, k)
while 1:
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'CPPN_confignew')
    pop = population.Population(config_path)
    pop.run(eval_fitness, 50001)
    if flag==1:
       clazz=(clazz+1)%8 #to avoid class 8 and 9 for now - change back to 10 remove later
    if clazz==3:          #dont do 3, 4 or 6
       clazz=clazz+1
    if clazz==4:
       clazz=clazz+1    
    if clazz==6:
        clazz=clazz+1
    if clazz==0:
        k=k+1
        clazz=clazz+1 #to avoid class 0 for not - remove later
