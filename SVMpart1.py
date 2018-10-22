import h5py
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn import svm

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
#plt.subplot(2,2,1)
plt.scatter(x1[:,0],x1[:,1], c=y1,cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('data1')
plt.close()
#plt.legend(loc='upper right')

plt.figure(2)
#plt.subplot(2,2,2)
plt.scatter(x2[:,0],x2[:,1], c=y2,cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('data2')
plt.close()

plt.figure(3)
#plt.subplot(2,2,3)
plt.scatter(x3[:,0],x3[:,1], c=y3,cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('data3')
plt.close()

plt.figure(4)
#plt.subplot(2,2,4)
plt.scatter(x4[:,0],x4[:,1], c=y4,cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('data4')
plt.close()




# create a mesh to plot
# x_min = x2[:, 0].min() - 1;
# x_max=x2[:, 0].max() + 1;
# y_min = x2[:, 1].min() - 1;
# y_max =x2[:, 1].max() + 1;
#
# h = 0.02 #step size in mesh
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h));
#
# # Plot the data for Proper Visual Representation
# clf = svm.SVC(kernel='linear',C=1)
# clf.fit(x2,y2);
#
# # Predict the result by giving Data to the model
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#
# plt.scatter(x2[:, 0], x2[:, 1], c=y2, cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.title('SVC with linear kernel')
##########################################################################################
def linear_kernel(x1, x2):
   # m=np.array([[2,0],[0,1.0]]);

    return np.dot(x1,x2.T)
clfl=svm.SVC(kernel=linear_kernel);
clfl.fit(x1,y1);






clf1=svm.SVC(kernel='linear');
clf2=svm.SVC(kernel='poly',degree=3);
clf3=svm.SVC(kernel='rbf',C=1);


def visualize(x,y,clf,data_name):
    x_min = x[:, 0].min() - 1;
    x_max = x[:, 0].max() + 1;
    y_min = x[:, 1].min() - 1;
    y_max = x[:, 1].max() + 1;

    h = 0.02  # step size in mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h));

    # Plot the data for Proper Visual Representation
    #clf = svm.SVC(kernel='linear',C=1)
    clf.fit(x, y);
    #print(clf.get_params());

    # Predict the result by giving Data to the model
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm,edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel'+data_name)

plt.figure(5)
#plt.subplot(,2,1)
visualize(x1,y1,clfl," (data-1)");

#plt.subplot(2,2,2)
plt.figure(6)
visualize(x2,y2,clfl," (data-2)");
#plt.subplot(2,2,3)
plt.figure(7)
visualize(x3,y3,clfl," (data-3)");
#plt.subplot(2,2,4)
plt.figure(8)
visualize(x4,y4,clfl," (data-4)");

plt.show()




###################################################################################################



def linear_kernel2(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))



clf=svm.SVC(kernel='linear')
clf.fit(x2,y2);
wieghts=clf.coef_;
dual_coeff=clf.dual_coef_;
svect=clf.support_vectors_;
indices_sv=clf.support_ ;
intercept=clf.intercept_ ;
print(wieghts)
print(dual_coeff)
print(svect)
print(indices_sv)

def _compute_kernel_support_vectors(x,sv):
    res = np.zeros((x.shape[0], sv.shape[0]))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(sv):
            res[i, j] = linear_kernel(x_i, x_j)
    return res

def predict(kernel_support_vectors,dual_coeff,intercept):
    #kernel_support_vectors = _compute_kernel_support_vectors(x)
    prod = np.multiply(kernel_support_vectors, dual_coeff)
    prediction = intercept + np.sum(prod, 1)
    return np.sign(prediction)

o1=_compute_kernel_support_vectors(x2,svect);
o2=predict(o1,dual_coeff,intercept);
#print(o2)
indices_y_positive = (y1 == 1);
#print(indices_y_positive)
indices_y_negative = (np.ones(x1[:,0].shape) - indices_y_positive).astype(bool);
#print(indices_y_negative)
postiveX=[]
negativeX=[]
for i,v in enumerate(y1):
    if v==0:
        negativeX.append(x1[i])
    else:
        postiveX.append(x1[i])
#print(postiveX);

def compute_wieghts(x,y):
    sample_size=x[:,0].shape;
    alpha=np.zeros(sample_size);
    gradient=np.ones(sample_size);






#for i, x_i in enumerate(x1):
  #  print(i,x_i)




