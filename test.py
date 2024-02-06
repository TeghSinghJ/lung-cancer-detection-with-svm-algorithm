import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
# Non-Binary Image Classification using Convolution Neural Networks
'''
path = 'Dataset'

labels = []
X = []
Y = []


for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            X.append(im2arr)
            if name == 'normal':
                Y.append(0)
            if name == 'abnormal':
                Y.append(1)
        
X = np.asarray(X)
Y = np.asarray(Y)
print(Y.shape)
print(X.shape)
print(Y)

X = X.astype('float32')
X = X/255
    
test = X[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
np.save('features/X.txt',X)
np.save('features/Y.txt',Y)
'''
X = np.load('features/X.txt.npy')
Y = np.load('features/Y.txt.npy')
X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))

pca = PCA(n_components = 100)
X = pca.fit_transform(X)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
cls = svm.SVC() 
cls.fit(X_train, y_train)
predict = cls.predict(X_test)
svm_acc = accuracy_score(y_test,predict)*100
print(svm_acc)

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_train)
predict = kmeans.predict(X_test)
acc = accuracy_score(y_test,predict)*100
print(acc)
print(kmeans.labels_)
centroids = kmeans.cluster_centers_
print(centroids)

cls = KNeighborsClassifier(n_neighbors = 2) 
cls.fit(X_train, y_train)
predict = cls.predict(X_test)
acc = accuracy_score(y_test,predict)*100

print(str(100-acc)+" "+str(100-svm_acc))

