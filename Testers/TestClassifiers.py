# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:57:48 2017

@author: Admin
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA


#%% Load data
os.chdir(r'D:\Agus\HiddenProject')
df = pd.read_pickle('Train.pandas')
#X = df[['hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7', 'hu8']]
cols = []
for j in range(1, 42):
    cols.append('S'+str(j))
X = df[cols]
y = df.c

classes = list(set(y))
count = Counter(df.c)
print(count)

#%% Define useful functions

# K fold function with its estimator
def KFold_test(clf, X, y):
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    classes = list(set(y))
    matrixes = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        
        matrix = conf_mat(y_test, y_predict, classes, Plot=False)
        matrixes.append(matrix)
    mean_mat = np.mean(matrixes, axis=0)
    std_mat = np.std(matrixes, axis=0)
    
    return mean_mat, std_mat

# Matrix functions
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
    
def conf_mat(y_correct, y_predict, classes, Plot=False):
    matrix = confusion_matrix(y_correct, y_predict, labels=classes)
    matrix = (matrix.T / matrix.T.sum(axis=0)).T
    
    if Plot:
        plot_conf_mat(matrix, classes)
    return matrix

#%% Test all classifiers and save PDF with results

def Test_and_savetoPDF(X, y, pdf_name):
    #Generate pdf file
    pp = PdfPages(pdf_name)
    
    #%% Cross validation for Decision Tree
    
    #scorer = make_scorer(confusion_matrix)
    mean_mat, std_mat = KFold_test(DecisionTreeClassifier(), X, y)
    print('Decision Tree Classifier')
    pp_plot_conf_mats(mean_mat, std_mat, classes, 'Decision Tree Classifier', pp)
    
    #%% Test K Nearest Neighbours
    
    neighborses = [1,2,3,4,5]
    radiuses = [10, 100, 1000]
    weights = ['uniform', 'distance']
    
    for weight in weights:
        for neighbors in neighborses:
            title = str(neighbors) + ' Nearest Neighbors weighted by ' + weight
            print(title)
            mean_mat, std_mat = KFold_test(KNeighborsClassifier(n_neighbors=neighbors, weights=weight), X, y)
            pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    for weight in weights:
        for radius in radiuses:
            try:
                title = str(radius) + ' Radius Neighbors weighted by ' + weight
                print(title)
                mean_mat, std_mat = KFold_test(RadiusNeighborsClassifier(radius=radius, weights=weight), X, y)
                pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
            except ValueError:
                continue
    
    #%% Test support vector machines
    
    Cs = [0.1, 0.2, 0.5, 1, 2, 5]
    gammas = [0.1, 0.2, 0.5, 1, 2, 5]
    
    for c in Cs:
        for gamma in gammas:
            title = 'SVM with c='+str(c)+' and $\gamma$='+str(gamma)
            print(title)
            mean_mat, std_mat = KFold_test(svm.SVC(kernel='rbf', C=c, gamma=gamma), X, y)
            pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    #%% Test Naive Bayes
    
    NBNames = ['Bernouilli Naive Bayes', 'Gaussian Naive Bayes']
    NBClassiffiers = [BernoulliNB(), GaussianNB()]
    
    for Name, NBClassiffier in zip(NBNames, NBClassiffiers):
        print(Name)
        mean_mat, std_mat = KFold_test(NBClassiffier, X, y)
        pp_plot_conf_mats(mean_mat, std_mat, classes, Name, pp)
    
    #%% Test Random Forest Classifier
    
    bootstraps = {'Bootstrap On':True, 'Bootstrap Off':False}
    n_trees = [10, 100, 1000]
    max_feats = np.asarray([.3, .5, .8])
    max_feats *= len(cols)
    max_feats = [int(x) for x in max_feats]
    
    for bootstrap_state, bootsrap_bool in bootstraps.items():
        for n_tree in n_trees:
            for max_feat in max_feats:
                title = 'Random Forest with '+str(n_tree)+' trees, '+str(max_feat)+' features and '+bootstrap_state
                print(title)
                clf = RandomForestClassifier(n_estimators=n_tree, max_features=max_feat, bootstrap=bootsrap_bool)
                mean_mat, std_mat = KFold_test(clf, X, y)
                pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    #%% Test Extra Trees Classifier
    
    bootstraps = {'Bootstrap On':True, 'Bootstrap Off':False}
    n_trees = [10, 100, 1000]
    max_feats = np.asarray([.3, .5, .8])
    max_feats *= len(cols)
    max_feats = [int(x) for x in max_feats]
    
    for bootstrap_state, bootsrap_bool in bootstraps.items():
        for n_tree in n_trees:
            for max_feat in max_feats:
                title = 'Extra Trees with '+str(n_tree)+' trees, '+str(max_feat)+' features and '+bootstrap_state
                print(title)
                clf = ExtraTreesClassifier(n_estimators=n_tree, max_features=max_feat, bootstrap=bootsrap_bool)
                mean_mat, std_mat = KFold_test(clf, X, y)
                pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    #%% Test Bagging of classifiers
    
    estimators = {'Decision Tree':DecisionTreeClassifier(), 
                  'K Nearest Neighbors':KNeighborsClassifier(n_neighbors=5, weights='distance'), 
                  'Radius Nearest Neghbors':RadiusNeighborsClassifier(radius=100, weights='distance'), 
                  'Support Vector Machine':svm.SVC(kernel='rbf'), 
                  'Bernoulli Naive Bayes':BernoulliNB()}
    bootstraps = {'Bootstrap On':True, 'Bootstrap Off':False}
    n_trees = [10, 100, 1000]
    max_feats = np.asarray([.3, .5, .8])
    max_feats *= len(cols)
    max_feats = [int(x) for x in max_feats]
    
    for estimator_name, estimator in estimators.items():
        for bootstrap_state, bootsrap_bool in bootstraps.items():
            for n_tree in n_trees:
                for max_feat in max_feats:
                    title = 'Bagging of '+estimator_name+' with '+str(n_tree)+' trees, '+str(max_feat)+' features and '+bootstrap_state
                    print(title)
                    clf = BaggingClassifier(base_estimator=estimator, n_estimators=n_tree, max_features=max_feat, bootstrap=bootsrap_bool)
                    mean_mat, std_mat = KFold_test(clf, X, y)
                    pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    #%% Test AdaBoost
    
    estimators = {'Decision Tree':DecisionTreeClassifier(), 
                  #'K Nearest Neighbors':KNeighborsClassifier(n_neighbors=5, weights='distance'), 
                  #'Radius Nearest Neghbors':RadiusNeighborsClassifier(radius=100, weights='distance'), 
                  'Support Vector Machine':svm.SVC(kernel='rbf'), 
                  'Bernoulli Naive Bayes':BernoulliNB()
                  }
    learn_rates = [0.1, 0.5, 0.8, 1]
    n_trees = [50, 100, 1000]
    
    for estimator_name, estimator in estimators.items():
        for n_tree in n_trees:
            for learn_rate in learn_rates:
                
                try:
                    title = 'Ada Boost of '+estimator_name+' with '+str(n_tree)+' trees, '+str(learn_rate)+' learn rate'
                    print(title)
                    clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=n_tree, learning_rate=learn_rate)
                    mean_mat,std_mat = KFold_test(clf, X, y)
                except TypeError:
                    title = 'Ada Boost of '+estimator_name+' with '+str(n_tree)+' trees, '+str(learn_rate)+' learn rate with SAMME'
                    print(title)
                    clf = AdaBoostClassifier(base_estimator=estimator, algorithm='SAMME', n_estimators=n_tree, learning_rate=learn_rate)
                    mean_mat,std_mat = KFold_test(clf, X, y)
                pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    #%% Test Gradient Boosting Classifier
    
    n_trees = [100, 500, 1000]
    max_feats = np.asarray([.3, .5, .8])
    max_feats *= len(cols)
    max_feats = [int(x) for x in max_feats]
    learn_rates = [0.1, 0.5, 0.8, 1]
    
    for n_tree in n_trees:
        for max_feat in max_feats:
            for learn_rate in learn_rates:
                title = 'Gradient Boost with '+str(n_tree)+' trees, '+str(learn_rate)+' learn rate and '+str(max_feat)+' features'
                print(title)
                clf = GradientBoostingClassifier(n_estimators=n_tree, max_features=max_feat, learning_rate=learn_rate)
                mean_mat, std_mat = KFold_test(clf, X, y)
                pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    #%% Test Voting Classifier
    
    estimator_tuples = [('Decision Tree', DecisionTreeClassifier()), 
                  ('K Nearest Neighbors',KNeighborsClassifier(n_neighbors=5, weights='distance')), 
                  ('Radius Nearest Neghbors',RadiusNeighborsClassifier(radius=100, weights='distance')), 
                  ('Support Vector Machine',svm.SVC(kernel='rbf')), 
                  ('Bernoulli Naive Bayes',BernoulliNB())
                  ]
    
    votes = ['hard']
    
    for vote in votes:
        estimators = ', '.join([name for name, _ in estimator_tuples])
        title = 'Voting Classifier with '+estimators+' estimators'
        print(title)
        clf = VotingClassifier(estimator_tuples, voting=vote)#, n_jobs=4)
        mean_mat, std_mat = KFold_test(clf, X, y)
        pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    
    estimator_tuples = [('Decision Tree', DecisionTreeClassifier()), 
                  ('K Nearest Neighbors',KNeighborsClassifier(n_neighbors=5, weights='distance')), 
                  #('Radius Nearest Neghbors',RadiusNeighborsClassifier(radius=100, weights='distance')), 
                  ]
    
    votes = ['hard']
    
    for vote in votes:
        estimators = ', '.join([name for name, _ in estimator_tuples])
        title = 'Voting Classifier with '+estimators+' estimators'
        print(title)
        clf = VotingClassifier(estimator_tuples, voting=vote)#, n_jobs=4)
        mean_mat, std_mat = KFold_test(clf, X, y)
        pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    #%% Test multiperceptron classifier
    
    title = 'MultiPerceptron Classifier'
    clf = MLPClassifier(max_iter=1000)
    print(title)
    mean_mat, std_mat = KFold_test(clf, X, y)
    pp_plot_conf_mats(mean_mat, std_mat, classes, title, pp)
    
    pp.close()

#%% Train all classifiers and save pdf
print('Test all without dimension transformation')
Test_and_savetoPDF(X, y, 'Classifiers_noRtCC_Zer.pdf')

#%% Repeat with PCA applied to data
print('Transform with PCA and reTest')
# train
pca = PCA()
X_new = X.copy()
#pca.fit(X)
X_new = pd.DataFrame(pca.fit_transform(X_new), index=X_new.index)

Test_and_savetoPDF(X_new, y, 'Classifiers_with_PCA_noRtCC.pdf')
#%% Apply iCA to data
print('Transform with ICA and reTest')
# train
ica = FastICA()
X_new = X.copy()
X_new = pd.DataFrame(ica.fit_transform(X_new), index=X_new.index)

Test_and_savetoPDF(X_new, y, 'Classifiers_with_ICA_noRtCC_Zer.pdf')