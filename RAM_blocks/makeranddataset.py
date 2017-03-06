from fuel.datasets.hdf5 import H5PYDataset, h5py
import numpy as np
import os 
import random

num_examples=3000
feat=np.zeros((num_examples,1,28,28))
targ=np.zeros((num_examples,1))

for i in range(0,num_examples):
    rando=np.random.ranf((28,28))
    cutoff=np.random.uniform(0.5,0.95)
    rando[rando<cutoff]=0
    rando[rando>=cutoff]=1
    feat[i,0,:,:] = rando
    targ[i,:] = random.randint(0,9)

f=h5py.File('randomnoise_{}.hdf5'.format(num_examples), mode='w')
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

split_dict = {'test': {'features': (0, num_examples), 'targets': (0, num_examples)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()
