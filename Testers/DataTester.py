# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:25:03 2017

@author: Admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio

from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold
#from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

#%% Plot image function

def plot_img(pos, pp):
    correct_label = y_test[pos]
    wrong_label = X_test.predict[pos]
    title = 'mixed '+correct_label+' for '+wrong_label+' in image '+str(pos)
    plt.imshow(skio.imread(df.img[pos]))
    plt.title(title)
    pp.savefig()
    plt.close()
    #dfc = df.copy()
    #dfc.c[pos] = 'VIEW'
    #q = 'c == "%s" or c == "%s" or c == "%s"' % (correct_label, wrong_label, 'VIEW')
    #sns.pairplot(dfc.query(q), hue='c', vars=df.columns[2:])

#%% Load data file

df = pd.read_pickle('Train.pandas')
X = df[df.columns[2:]]
y = df.c

classes = list(set(df.c))

#%% Transform with PCA

print('Transform with PCA')
# train
pca = PCA()
X_new = X.copy()
#pca.fit(X)
X_new = pd.DataFrame(pca.fit_transform(X_new), index=X_new.index)

#%% Separate in training and testing groups

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3)

#%% Select classifier and train it

count = Counter(y_train.values)
print(count)
#clf = RadiusNeighborsClassifier(radius=100, weights='distance')
#clf = KNeighborsClassifier(n_neighbors=10, weights='distance')

n_tree= 1000
max_feat = 3
bootsrap_bool = True
clf = ExtraTreesClassifier(n_estimators=n_tree, max_features=max_feat, bootstrap=bootsrap_bool)
clf.fit(X_train, y_train)

#%% Check predictability over training data

X_train['predict'] = clf.predict(X_train)
X_train['correct'] = X_train.predict.values==y_train.values

acquracy = np.sum(X_train['correct'])/len(X_train['correct'])
print('acquracy over training set: '+str(acquracy))

#%% Predict and test over testing data

X_test['predict'] = clf.predict(X_test)
X_test['correct'] = X_test.predict.values==y_test.values

acquracy = np.sum(X_test['correct'])/len(X_test['correct'])
print('acquracy over test set: '+str(acquracy))

#%% types of errors

pairs = []
print('amount of errors:'+ str(len(X_test)-np.sum(X_test['correct'])))
print('Errors commited:')
pp = PdfPages('img_PCA_ExtraTrees.pdf')
for i in X_test.index:
    if not X_test.correct[i]:
        #print('mixed '+y_test[i]+' for '+X_test.predict[i]+' in image '+str(i))
        pairs.append((y_test[i], X_test.predict[i]))
        plot_img(i, pp)
        
matrix = confusion_matrix(y_test, X_test.predict, labels=classes)
plt.matshow((matrix.T / matrix.T.sum(axis=0)).T)
plt.colorbar()
plt.xticks(list(range(len(classes))),classes)
