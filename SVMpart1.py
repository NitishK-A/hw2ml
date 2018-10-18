import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

filename1='data_1.h5'
filename2='data_2.h5'
filename3='data_3.h5'
filename4='data_4.h5'

f1=h5py.File(filename1,'r');
f2=h5py.File(filename2,'r');
f3=h5py.File(filename3,'r');
f4=h5py.File(filename4,'r');



def insidehdf5(file):

    print("Name of the dataset file -->"+file.filename)
    #    print(file.shape)
    print("Keys in the dataset -->"+str(list(file)))

    key1 = list(file.keys())[0];
    print("Shape of x-->"+str(file[key1].shape))
    print("Values in "+ str(key1[0]))
    x=(file[key1][:]);
    print(file[key1][:]);

    key2 = list(file.keys())[1];
    print("Shape of y-->" + str(file[key2].shape))
    #print("Values in " + str(key2[0]))
    y=(file[key2][:]);
    print(file[key2][:]);
    return x,y





x1,y1=insidehdf5(f1);
x2,y2=insidehdf5(f2);
x3,y3=insidehdf5(f3);
x4,y4=insidehdf5(f4);

#print(x1);
#print(y1);

plt.figure(1)
#plt.plot(x, 'o',color='b',label='x')
#plt.plot(y, 'o',color='m',label='y')
#plt.scatter(np.arange(100),x[:,0],color='r');
#plt.scatter(np.arange(100),x[:,1],color='y');
plt.subplot(2,2,1)
plt.scatter(x1[:,0],x1[:,1], c=y1,cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('data1')
#plt.legend(loc='upper right')

plt.subplot(2,2,2)
plt.scatter(x2[:,0],x2[:,1], c=y2,cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('data2')

plt.subplot(2,2,3)
plt.scatter(x3[:,0],x3[:,1], c=y3,cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('data3')

plt.subplot(2,2,4)
plt.scatter(x4[:,0],x4[:,1], c=y4,cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('data4')

plt.show()



