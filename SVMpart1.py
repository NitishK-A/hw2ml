import h5py
import numpy as np

filename1='data_1.h5'
filename2='data_2.h5'
filename3='data_3.h5'
filename4='data_4.h5'

f1=h5py.File(filename1,'r');



def insidehdf5(file):

    print("Name of the dataset file -->"+file.filename)
#    print(file.shape)
    print("Seys in the dataset -->"+str(list(file)))

    key1 = list(file.keys())[0];
    print("Shape of x-->"+str(file[key1].shape))
    print("Values in "+ str(key1[0]))
    print(file[key1][:]);

    key2 = list(file.keys())[1];
    print("Shape of y-->" + str(file[key2].shape))
    print("Values in " + str(key2[0]))
    print(file[key2][:]);


insidehdf5(f1)




