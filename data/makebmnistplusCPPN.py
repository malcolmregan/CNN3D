#make array the size of bmnist+CPPNexamples
#open bmnist
#iterate over dataset put into the new dataset (except for class 1)
#iterate over CPPN examples save as target=1
#make fuel
#train lenet 50 epochs on this dataest

from fuel.datasets.hdf5 import H5PYDataset, h5py
import numpy as np
import os
import random
import math

train_set = H5PYDataset('../data/bmnistplusCPPN.hdf5', which_sets=('train',))
handle = train_set.open()
test_set = H5PYDataset('../data/bmnistplusCPPN.hdf5', which_sets=('test',))
handle2 = test_set.open()

num=0
datapath='../RAM_blocks/CPPNGenerated/LeNet/moreefficient'
for i in range(0,10):
    path=os.path.join(datapath, 'class{}'.format(i))
    num=num+len(os.walk(path).next()[2])
newCPPNnum=num
num=num+train_set.num_examples+test_set.num_examples

print "Number of CPPN Examples: {}".format(newCPPNnum)
print "Total size of new dataset: {}".format(num)

idxs=[0]*num
for i in range(0,num):
    idxs[i]=i
random.shuffle(idxs)

np.set_printoptions(threshold=np.nan)
feat=np.zeros((num,1,28,28))
targ=np.zeros((num,1))

raw_input()

print("Processing CPPN data...")
i=0
for x in range(0,10):
    path=os.path.join(datapath, 'class{}'.format(x))    
    for files in os.listdir(path):
        if files.endswith(".npz"):
            example=np.load(os.path.join(path,files))
            feat[idxs[i],:,:,:]=example['features'][0]
            targ[idxs[i],:]=10 #noise class
            i=i+1

print("Processing BMNIST training set...")
k=0
for examples in range(0, train_set.num_examples):
    data=train_set.get_data(handle, slice(examples, examples+1))
    feat[(idxs[examples+i]),:,:,:]=data[0]
    targ[(idxs[examples+i]),:]=data[1][0][0]
    k=k+1

print("Processing BMNIST test set...")
h=0
for examples in range(0, test_set.num_examples):
    data=test_set.get_data(handle2, slice(examples, examples+1))
    feat[(idxs[examples+k+i]),:,:,:]=data[0]
    targ[(idxs[examples+k+i]),:]=data[1][0][0]
    h=h+1

print("Creating bmnistplusCPPN2.hdf5...")
f=h5py.File('bmnistplusCPPN2.hdf5', mode='w')
features=f.create_dataset('features',np.shape(feat),dtype='uint8')
features[...]=feat
targets=f.create_dataset('targets',np.shape(targ),dtype='uint8')
targets[...]=targ

features.dims[0].label='batch'
features.dims[1].label='input'
features.dims[2].label=''
features.dims[3].label=''
targets.dims[0].label='batch'
targets.dims[1].label='index'

split_dict = {'train': {'features': (0, int(math.floor(num*.86))), 'targets': (0, int(math.floor(num*.86)))}, 'test': {'features': (int(math.floor(num*.86)), num), 'targets': (int(math.floor(num*.86)), num)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()

raw_input()

print("Checking dataset...")
train_set = H5PYDataset('../data/bmnistplusCPPN2.hdf5', which_sets=('train',))
handle = train_set.open()
test_set = H5PYDataset('../data/bmnistplusCPPN2.hdf5', which_sets=('test',))
handle2 = test_set.open()

i=0
for examples in range(0, train_set.num_examples):
    data=train_set.get_data(handle, slice(examples, examples+1))
    if np.sum(data[0])==0:
        i=i+1

for examples in range(0, test_set.num_examples):
    data=test_set.get_data(handle2, slice(examples, examples+1))
    if np.sum(data[0])==0:
        i=i+1

print "Number of empty examples: {}".format(i)
print "Done."
