# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 13:40:19 2021

@author: Anshul
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


#%%

category = ['rec.sport.hockey', 'sci.electronics','rec.autos']
newsgroups_train = fetch_20newsgroups(subset = 'train', categories = category)
newsgroups_test = fetch_20newsgroups(subset = 'test',  categories = category)

#Print categories names
print(list(newsgroups_train.target_names)) # ----> Name of classes ['rec.autos', 'rec.sport.hockey', 'sci.electronics']

#Printing shape of data and target
#print(newsgroups_train.filenames.shape) ----> (1785,) 
#print(newsgroups_train.target.shape)    ----> (1785,) 
print(np.unique(newsgroups_train.target)) # ----> [0,1,2]

print("-------- TRAIN DATA ---------")
print("Before Preprocessing : ")
print(newsgroups_train.data[2]) 
print(newsgroups_train.target[2])

#%%
#Cleaning Texts and preprocessing
lemmatizer = WordNetLemmatizer()

data_length = len(newsgroups_train.data)

corpus = []
for i in range(data_length):
    review = re.sub('[^a-zA-z0-9]',' ',newsgroups_train.data[i])
    review = review.lower() #Lowering the words in sentence
    review = review.split() #Splitting the sentences in words 
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#%%

print("After cleaning and Preprocessing : ")
print(np.array(corpus).shape)    
print(corpus[2])


#%%

# Creating TFIDF Model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()

X = cv.fit_transform(corpus).toarray() #make matrix according to their frequencies in a sentence.
print(X.shape)

#Validation data
from sklearn.model_selection import train_test_split

X_train, validation_data, y_train, validation_target = train_test_split(X, newsgroups_train.target, test_size= 0.1, random_state= 1234)

#%%
# =============================================================================
#
# --------------- --------------Test Data -------------------------------------
# 
# =============================================================================


print("-------- TEST DATA ---------")
# print("Before Preprocessing : ")
# print("Data : ", newsgroups_test.data[2])
# print("Class : ", newsgroups_test.target[2])

test_data_length = len(newsgroups_test.data)

test_corpus = []
for i in range(test_data_length):
    t_review = re.sub('[^a-zA-z0-9]',' ',newsgroups_test.data[i])
    t_review = t_review.lower() #Lowering the words in sentence
    t_review = t_review.split() #Splitting the sentences in words 
    t_review = [lemmatizer.lemmatize(word) for word in t_review if word not in set(stopwords.words('english'))]
    t_review = ' '.join(t_review)
    test_corpus.append(t_review)


#TF-IDF on Test Data
X_test_data = cv.transform(test_corpus).toarray() #make matrix according to their frequencies in a sentence.
print("Tfidf Test data Matrix Shape: ", X_test_data.shape)

#%%
# =============================================================================
# #---------------------------Classifier Model---------------------------------
# =============================================================================

#Classifier Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

best_validation_f1score = 0
best_test_f1score = 0

classifiers = ['NaiveBayes','DecisionTree', 'SVM', 'RandomForest', 'AdaBoost', 'MLP']
# classifiers = ['MLP']

for classifier in classifiers:
    if classifier == 'NaiveBayes':
        clf = MultinomialNB(alpha = 0.05).fit(X_train, y_train)
        clf_validation_pred = clf.predict(validation_data)
        clf_test_pred = clf.predict(X_test_data)
        
    elif classifier == 'DecisionTree':
        clf = DecisionTreeClassifier().fit(X_train, y_train)
        clf_validation_pred = clf.predict(validation_data)
        clf_test_pred = clf.predict(X_test_data)
        
    elif classifier == 'SVM':
        clf = SVC().fit(X_train, y_train)
        clf_validation_pred = clf.predict(validation_data)
        clf_test_pred = clf.predict(X_test_data)
        
    elif classifier == 'RandomForest':
        clf = RandomForestClassifier().fit(X_train, y_train)
        clf_validation_pred = clf.predict(validation_data)
        clf_test_pred = clf.predict(X_test_data)
        
    elif classifier == 'AdaBoost':
        clf = AdaBoostClassifier().fit(X_train, y_train)
        clf_validation_pred = clf.predict(validation_data)
        clf_test_pred = clf.predict(X_test_data)
        
    elif classifier == 'MLP':
        clf = MLPClassifier(alpha = 0.1).fit(X_train, y_train)
        clf_validation_pred = clf.predict(validation_data)
        clf_test_pred = clf.predict(X_test_data)
    
    print("Results of classifier : {}".format(classifier))
    
    #Accuracy, confusion matrix and F1 score on Validation Data
    confusion = metrics.confusion_matrix(validation_target, clf_validation_pred)
    accuracy = metrics.accuracy_score(validation_target, clf_validation_pred)
    f_score = metrics.f1_score(validation_target, clf_validation_pred, average= 'micro')
    
    print("Confusion Matrix of validation  : ", confusion)
    print("Accuracy of Model on validation : ", accuracy )
    print("F1 Score of model on validation : ", f_score)
    
    
    
    #Accuracy, confusion matrix and F1 score on Test Data
    test_confusion = metrics.confusion_matrix(newsgroups_test.target, clf_test_pred)
    test_accuracy = metrics.accuracy_score(newsgroups_test.target, clf_test_pred)
    test_f_score = metrics.f1_score(newsgroups_test.target, clf_test_pred, average= 'micro')
    
    print("Confusion Matrix of Test  : ", test_confusion)
    print("Accuracy of Model on Test : ", test_accuracy)
    print("F1 Score of model on Test : ", test_f_score)
    
    
    if best_test_f1score < test_f_score:
        best_validation_f1score = f_score
        best_test_f1score = test_f_score
        Model = classifier
        
    print(" --------------xxxxxxxxxxxxx---------------")

#%%
#Final Results

print("Best Model : {}".format(Model))
print("Models Validation F1 score : {}".format(best_validation_f1score))
print("Models Test F1 score : {}".format(best_test_f1score))

#%%

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2).fit_transform(X)
print(tsne.shape)

import matplotlib.pyplot as plt

plt.scatter(tsne[:,0], tsne[:,1], c = newsgroups_train.target)


