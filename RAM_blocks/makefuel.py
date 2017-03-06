from fuel.datasets.hdf5 import H5PYDataset, h5py
import numpy as np
import os 

#model='RAM'
model='LeNet'
#model='Draw'

for CLASS in range(0,10):
    datapath='./CPPNGenerated/{0}/class{1}'.format(model, CLASS)
    num=len(os.walk(datapath).next()[2])

    np.set_printoptions(threshold=np.nan)
    feat=np.zeros((num,1,28,28))
    targ=np.zeros((num,1))

    for files in os.listdir(datapath):
        if files.endswith(".npz"):
            print(files)
            fileindex=int((str(files).split('_')[1]).split('.')[0])
            example=np.load(os.path.join(datapath,files))
            feat[fileindex,:,:,:]=example['features'][0]
            targ[fileindex,:]=example['targets'][0]

    f=h5py.File('{0}_CLASS{1}.hdf5'.format(model, CLASS, (num)), mode='w')
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

    split_dict = {'test': {'features': (0, num), 'targets': (0, num)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()
