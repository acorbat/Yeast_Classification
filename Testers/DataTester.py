# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:25:03 2017

@author: Admin
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio

from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
#from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

os.chdir(r'D:\Agus\HiddenProject')
#%% Plot image function

# Matrix functions
def KFold_test(clf, X, y, classes=None):
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    if classes is None:
        classes = list(set(y))
    matrixes_un = []
    matrixes = []
    matrixes_FPR = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        
        matrix_un = conf_mat(y_test, y_predict, classes, Plot=False, this_axis=None)
        matrix = conf_mat(y_test, y_predict, classes, Plot=False)
        matrix_FPR = conf_mat(y_test, y_predict, classes, Plot=False, this_axis=0)
        matrixes_un.append(matrix_un)
        matrixes.append(matrix)
        matrixes_FPR.append(matrix_FPR)
    mean_mat_un = np.mean(matrixes_un, axis=0)
    mean_mat = np.mean(matrixes, axis=0)
    std_mat = np.std(matrixes, axis=0)
    mean_mat_FPR = np.mean(matrixes_FPR, axis=0)
    std_mat_FPR = np.std(matrixes_FPR, axis=0)
    
    plot_table(mean_mat_un, classes)
    
    return mean_mat, std_mat, mean_mat_FPR, std_mat_FPR

def plot_conf_mat(mat, classes):
    fig1, ax = plt.subplots()
    ax.imshow(mat, interpolation ='none', cmap='BuGn')
    for (j,i),label in np.ndenumerate(mat):
        label = '%.2f' % label
        ax.text(i,j,label,ha='center',va='center')
        
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(list(range(len(classes))),classes)
    plt.yticks(list(range(len(classes))),classes)

def pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp):
    fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(11,5))
    
    # Plot confusion matrix
    ax1.imshow(mean_mat, interpolation ='none', cmap='BuGn')
    for (j,i),label in np.ndenumerate(mean_mat):
        label = '%.2f' % label
        ax1.text(i,j,label,ha='center',va='center')
    
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    ax1.set_xticks(list(range(len(classes))),classes)
    ax1.set_yticks(list(range(len(classes))),classes)
    
    # Plot standard deviaton of confusion matrix
    
    ax2.imshow(std_mat, interpolation ='none', cmap='BuGn')
    for (j,i),label in np.ndenumerate(std_mat):
        label = '%.2f' % label
        ax2.text(i,j,label,ha='center',va='center')
    
    ax2.set_title('STD Confusion Matrix')
    ax2.set_xlabel('Predicted label')
    ax2.set_xticks(list(range(len(classes))),classes)
    ax2.set_yticks(list(range(len(classes))),classes)
    
    plt.setp((ax1, ax2), xticks=list(range(len(classes))), xticklabels=classes, yticks=list(range(len(classes))), yticklabels=classes)
    plt.suptitle(title)
    pp.savefig()
    plt.close()
    
def conf_mat(y_correct, y_predict, classes, Plot=False, this_axis=1):
    """
    this_axix = 0 -> rows are normalized so one can see False Positive Rate
    this_axix = 1 -> cols are normalized so one can see True Positive Rate
    """
    matrix = confusion_matrix(y_correct, y_predict, labels=classes)
    if this_axis is not None:
        matrix = normalize(matrix, axis=this_axis, norm='l1')
    
    if Plot:
        plot_conf_mat(matrix, classes)
    return matrix

def plot_img(pos, ax):
    correct_label = y_test[pos]
    wrong_label = X_test.predict[pos]
    title = 'True: '+correct_label+'; Predicted: '+wrong_label+'; in image '+str(pos)
    ax.imshow(skio.imread(df.img[pos]))
    plt.suptitle(title)
    #dfc = df.copy()
    #dfc.c[pos] = 'VIEW'
    #q = 'c == "%s" or c == "%s" or c == "%s"' % (correct_label, wrong_label, 'VIEW')
    #sns.pairplot(dfc.query(q), hue='c', vars=df.columns[2:])

def plot_proba(pos, axs):
    probas = clf.predict_proba(X_test.loc[pos, range(len(cols))])
    this_classes = clf.classes_
    axs.axis('tight')
    axs.axis('off')
    axs.table(cellText=probas,
              rowLabels=['Probability'],
              colLabels=this_classes,
              loc='center')
    
def plot_mistake(pos, pp):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    plot_img(pos, axes[0])
    plot_proba(pos, axes[1])
    pp.savefig()
    plt.close()

def gen_tmat(TP, TN, FP, FN):
    mat = np.zeros((3, 3))
    mat[0,0] = TP
    mat[0,1] = FN
    mat[1,0] = FP
    mat[1,1] = TN
    mat[0,2] = TP/(TP+FP)
    mat[2,0] = TP/(TP+FN)
    mat[1,2] = FP/(FP+TN)
    mat[2,1] = FN/(FN+TN)
    mat[2,2] = np.nan
    return mat

def plot_tmat(mat, this_class, ax):
    ax.imshow(mat, interpolation ='none', cmap='BuGn')
    for (j,i),label in np.ndenumerate(mat):
        label = '%.2f' % label
        ax.text(i,j,label,ha='center',va='center')
    ax.set_title(this_class)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    
def plot_and_savemats(matrix, classes):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 6), sharex=True, sharey=True)
    for i, (this_class, this_ax) in enumerate(zip(classes, axes.flatten())):
        TP = matrix[i, i]
        FN = np.sum([matrix[i, j] for j in range(matrix.shape[1]) if j!=i])
        FP = np.sum([matrix[j, i] for j in range(matrix.shape[0]) if j!=i])
        TN = np.sum([matrix[j, k] for k in range(matrix.shape[1]) for j in range(matrix.shape[0]) if j!=i and k!=i])
        
        this_mat = gen_tmat(TP, TN, FP, FN)
        plot_tmat(this_mat, this_class, this_ax)
    plt.suptitle('One vs. All confusion matrixes')
    plt.setp(axes, xticks=list(range(2)), xticklabels=['True', 'False'],
        yticks=list(range(2)), yticklabels=['True', 'False'])
    plt.tight_layout()
    pp.savefig()
    plt.close()
    
def plot_table(matrix, classes):
    els = ((0,2), (2,0), (1,2), (2,1))
    table = []
    fig, axs = plt.subplots(1,1)
    for i, this_class in enumerate(classes):
        TP = matrix[i, i]
        FN = np.sum([matrix[i, j] for j in range(matrix.shape[1]) if j!=i])
        FP = np.sum([matrix[j, i] for j in range(matrix.shape[0]) if j!=i])
        TN = np.sum([matrix[j, k] for k in range(matrix.shape[1]) for j in range(matrix.shape[0]) if j!=i and k!=i])
        
        this_mat = gen_tmat(TP, TN, FP, FN)
       
        table.append(['%.2f' % this_mat[r, c] for (r, c) in els])

    axs.axis('tight')
    axs.axis('off')
    axs.table(cellText=table,
              rowLabels=classes,
              colLabels=['Positive Predicted Value', 'True Positive Rate', 'False Positive Rate', 'Negative Predictive Value'],
              loc='center')

    plt.title('Extra Trees with 1000 estimators, 3 features and bootstrap on')
    pp.savefig()
    plt.close()

#%% Load data file

df = pd.read_pickle('Train.pandas')

#%% Prefilter some data corrected
Remove_deform = False
if Remove_deform:
    for i in df.index:
        # Remove deform cells
        if df['c'][i]=='deform':
            df = df.drop(i)
            continue
    """
    # Correct wrong labeled cases
    possibles = corrections['Según máscara de Cellid'][i].split('/')
    predicted = corrections['Predicted'][i]
    if corrections[corrections.columns[2]][i] not in possibles:
        correct = predicted if predicted in possibles else possibles[0]
        df.set_value(corrections['image'][i], 'c', correct)
        continue
"""
#%% Separate data in features and classes

#cols = ['hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7', 'hu8']
cols = []
for j in range(1, 42):
    cols.append('S'+str(j))
cols = cols[:8]
X = df[cols]
y = df.c

classes = list(set(df.c))
classes.sort()

#%% Transform with PCA

print('Transform with PCA')
# train
pca = PCA()
X_new = X.copy()
#pca.fit(X)
X_new = pd.DataFrame(pca.fit_transform(X_new), index=X_new.index)

#%% Select classifier

#clf = RadiusNeighborsClassifier(radius=100, weights='distance')
#clf = KNeighborsClassifier(n_neighbors=10, weights='distance')

n_tree= 1000
max_feat = 3
bootsrap_bool = True
clf = ExtraTreesClassifier(n_estimators=n_tree, max_features=max_feat, bootstrap=bootsrap_bool)

#%% Generate confusion matrixes

if Remove_deform:
    pp = PdfPages('PCA_ExtraTrees_probas_sindeform_noRtCC_Zer8.pdf')
else:
    pp = PdfPages('PCA_ExtraTrees_probas_condeform_noRtCC_Zer8.pdf')

mean_mat, std_mat, mean_mat_FPR, std_mat_FPR = KFold_test(clf, X_new, y, classes=classes)
title = 'Confusion Matrix normalized for True Label'
pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)

title = 'Confusion Matrix normalized for Predicted Label'
pp_plot_conf_mats(mean_mat_FPR, std_mat_FPR, classes, title, pp)

#%% Separate in training and testing groups

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3)

#%% Train classifier

count = Counter(y_train.values)
pp.attach_note(count)

clf.fit(X_train, y_train)

#%% Check predictability over training data

X_train['predict'] = clf.predict(X_train)
X_train['correct'] = X_train.predict.values==y_train.values

#acquracy = np.sum(X_train['correct'])/len(X_train['correct'])
#print('acquracy over training set: '+str(acquracy))

#%% Predict and test over testing data

X_test['predict'] = clf.predict(X_test)
X_test['correct'] = X_test.predict.values==y_test.values

acquracy = np.sum(X_test['correct'])/len(X_test['correct'])
pp.attach_note('acquracy over test set: '+str(acquracy))

#%% types of errors

pairs = []
pp.attach_note('amount of errors:'+ str(len(X_test)-np.sum(X_test['correct'])))
pp.attach_note('Errors commited:')


os.chdir(r'D:\Agus\HiddenProject\Send')
for i in X_test.index:
    if not X_test.correct[i]:
        #print('mixed '+y_test[i]+' for '+X_test.predict[i]+' in image '+str(i))
        pairs.append((y_test[i], X_test.predict[i]))
        plot_mistake(i, pp)

for i in X_test.index:
    if X_test.correct[i]:
        plot_mistake(i, pp) # Actually not a mistake
        
#matrix = confusion_matrix(y_test, X_test.predict, labels=classes)
#plt.matshow((matrix.T / matrix.T.sum(axis=0)).T)
#plt.colorbar()
#plt.xticks(list(range(len(classes))),classes)

#%% Close pdf

pp.close()