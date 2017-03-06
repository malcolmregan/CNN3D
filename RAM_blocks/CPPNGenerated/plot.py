import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import math

def plotarray(data):
    string=str(data)
    print string
    data=np.load(data)
    array=data['features']
    array=array[0,0,:,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(array, cmap='gray_r', interpolation='none')

    axes = plt.gca()
    axes.set_xlim([0,27])
    axes.set_ylim([0,27])

    plt.title(string)

    plt.show()

def plotsave(data):
    string=str(data)
    string=string.split('.')[0]
    print string

    data=np.load(data)
    array=data['features']
    array=array[0,0,:,:]

    filled=np.nonzero(array>0)
    x=filled[1]
    y=filled[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)

    axes = plt.gca()
    axes.set_xlim([0,28])
    axes.set_ylim([0,28])

    plt.savefig(string.split('/')[0]+'/images/'+string.split('/')[1])
    plt.close()

def plotclass(CLASS):
    
    #model='RAM'
    model='LeNet'
    #model='Draw'
   
    fig = plt.figure()

    datapath = '{0}/moreefficient/class{1}'.format(model,CLASS)
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
        rowlength=4
    elif numfiles>=17 and numfiles<21:
        rowlength=5
    elif numfiles>=21 and numfiles<25:
        rowlength=6
    elif numfiles>=25 and numfiles<29:
        rowlength=7
    else: 
        rowlength=8

    for files in os.listdir(datapath): 
        string=str(files)
        string=string.split('.')[0]
        fileindex=int(string.split('_')[1])
        print string
        data=np.load(os.path.join(datapath, files))
        array=data['features']
        array=array[0,0,:,:]
        filled=np.nonzero(array>0)
        x=filled[1]
        y=filled[0]
         
        ax[fileindex] = fig.add_subplot(int(math.ceil(float(numfiles)/rowlength)),rowlength,fileindex+1, aspect='equal')
        ax[fileindex].scatter(x, y)

        axes = plt.gca()
        axes.set_xlim([0,28])
        axes.set_ylim([0,28])
        
        ax[fileindex].set_title(string)       
    """  
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    """ 
 
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.suptitle('{0} CPPN, Class {1}'.format(model, CLASS), size=25) 
    plt.show()
    #plt.savefig('class{}'.format(CLASS))
    plt.close()

def saveall():
    for i in range(0,10):
        path='class{}'.format(i)
        for files in os.listdir(path):
            if files.endswith(".npz"):
                plotsave(os.path.join(path,files))
