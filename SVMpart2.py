import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn import svm
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




train_labels = os.listdir('/Users/nitishatal/PycharmProjects/ML_hw2/Train_val');
train_labels.sort();
print("These are the following labels : ");
print(train_labels);
print();




def Hu_Moments(image):#reference from -https://gogul09.github.io/software/image-classification-python
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image))
    return feature.flatten()

def hog(image):#reference from -https://gogul09.github.io/software/image-classification-python
    bins=8;
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    o=hist.flatten()
    return o



labels1 = []
global_features1 = []
i,j,k,l,m,n=0,0,0,0,0,0;

for name in train_labels:
    path=os.path.join('/Users/nitishatal/PycharmProjects/ML_hw2/Train_val',name);
    current_label = name;
    for img in glob.glob("/Users/nitishatal/PycharmProjects/ML_hw2/Train_val"+"/"+name+"/*.png"):

        image = cv2.imread(img)
        image = cv2.resize(image, (500, 500));
        if1 = Hu_Moments(image);
        if2 = hog(image);
        global_feature = np.hstack([if2,if1]);
        #h=hog(image);
        global_features1.append(global_feature);
        #labels.append(current_label)
        if(name=='character_1_ka'):
            i=i+1;
            labels1.append(1)
        if (name == 'character_2_kha'):
            j = j + 1;
            labels1.append(2)
        if (name == 'character_3_ga'):
            k = k + 1;
            labels1.append(3)
        if (name == 'character_4_gha'):
            l = l + 1;
            labels1.append(4)
        if (name == 'character_5_kna'):
            m = m + 1;
            labels1.append(5)
    n=n+1;

print("(Training set)No. of labels :"+str(n));
print("(Training set)No. of images in each labels respectively : ");
print(i,j,k,l,m);
print()




labels2 = [];
global_features2 = [];
i,j,k,l,m,n=0,0,0,0,0,0;

for name2 in train_labels:
    print(name2)
    path = os.path.join('/Users/nitishatal/PycharmProjects/ML_hw2/Test', name2);
    current_label = name2;
    for img2 in glob.glob("/Users/nitishatal/PycharmProjects/ML_hw2/Test" + "/" + name2 + "/*.png"):

        image2 = cv2.imread(img2);
        image2 = cv2.resize(image2, (500, 500));
        fv_hu_moments2 = Hu_Moments(image2);
        fv_histogram = hog(image2);
        global_feature2 = np.hstack([fv_histogram,fv_hu_moments2]);
        #h2 = hog(image2);
        global_features2.append(global_feature2);
        # labels.append(current_label)
        if (name2 == 'character_1_ka'):
            i = i + 1;
            labels2.append(1)
        if (name2 == 'character_2_kha'):
            j = j + 1;
            labels2.append(2)
        if (name2 == 'character_3_ga'):
            k = k + 1;
            labels2.append(3)
        if (name2 == 'character_4_gha'):
            l = l + 1;
            labels2.append(4)
        if (name2 == 'character_5_kna'):
            m = m + 1;
            labels2.append(5)
    n = n + 1;

print("(Test set)No. of images in each labels respectively : ")
print(i,j,k,l,m);
print()
# clf = svm.SVC()
# labels =  np.array(train_labels).reshape(len(train_labels),1)
# hog_features = np.array(global_features);
# data_frame = np.hstack((hog_features,labels))
# percentage = 80
# partition = int(len(hog_features)*percentage/100)
# x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
# y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
#
# clf.fit(x_train,y_train)



trainfeatures1 = np.array(global_features1);
trainlabels1 = np.array(labels1);
print("shape :")
print(trainfeatures1.shape);

trainfeatures2 = np.array(global_features2);
trainlabels2 = np.array(labels2);

odata=np.array(trainfeatures2);
print('Shapetest'+str(odata.shape));
olabel=np.array(trainlabels2);
print('Shapetestlabel'+str(olabel.shape));



(trainData, validationData, trainLabels, validationLabels) = train_test_split(np.array(trainfeatures1), np.array(trainlabels1),test_size=0.20)

print ("splitted train and Validation set...")
print ("Train data  : {}".format(trainData.shape))
print ("Validation data   : {}".format(validationData.shape))
print ("Train labels: {}".format(trainLabels.shape))
print ("Validation labels : {}".format(validationLabels.shape))
print()

# Grid Search
# Parameter Grid
#param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
#param_grid = {'C': [0.1, 1], 'gamma': [1, 0.1]}

clf=svm.SVC(kernel='linear',C=100,gamma=10); #svm with rbf kernel
# Make grid search classifier
#clf = GridSearchCV(svm.SVC(kernel='rbf'),param_grid, verbose=1)
print(clf)
clf.fit(trainData,trainLabels); # fitting our model
#print("Best Parameters:\n", clf.best_params_)
#print("Best Estimators:\n", clf.best_estimator_)

y_pred_train=clf.predict(trainData);
training_acc=accuracy_score(trainLabels,y_pred_train);
s_t=clf.score(global_features2,labels2)
print(s_t);
print("Training Accuracy  : {}".format(training_acc));
#print(classification_report(y_test, y_pred))
print()

y_pred_val = clf.predict(validationData);
validation_acc= (accuracy_score(validationLabels, y_pred_val));
print("Validation Accuracy  : {}".format(validation_acc));
print()

y_pred_test=clf.predict(odata);
test_acc=accuracy_score(olabel,y_pred_test);
print("Test Accuracy  : {}".format(test_acc));
print()

# confusion Matrix for test set:
cmatrix=confusion_matrix(olabel,y_pred_test);
print(cmatrix)



