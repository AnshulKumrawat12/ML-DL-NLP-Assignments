# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 11:26:32 2021

@author: Anshul
"""

import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import gensim
import re # Regular Expression
import nltk
from nltk.stem import WordNetLemmatizer
import contractions
#%%
#---------------------------Load Data----------------------------

PATH_POS = 'D:/IISc/IISc 3rd Semester/DLNLP/Assignments/Assignment 3/Data/rt-polaritydata/rt-polarity.pos'
PATH_NEG = 'D:/IISc/IISc 3rd Semester/DLNLP/Assignments/Assignment 3/Data/rt-polaritydata/rt-polarity.neg'

Data_pos = []
Data_neg = []        
        
def load_data(Path):
    temp = []
    with open(Path, 'r', errors= 'ignore') as f:
        for line in f:
            temp.append(line)
    
    return temp

Data_pos_x = load_data(PATH_POS)
Data_neg_x = load_data(PATH_NEG)
Data_pos_y = list(np.ones(5331, dtype = 'int32'))
Data_neg_y = list(np.zeros(5331, dtype = 'int32'))

#%%

##--------------Splitting data into test and train-----------------------------

X_pos, X_test_pos, y_pos, y_test_pos = train_test_split(Data_pos_x, Data_pos_y, test_size= 831, shuffle = False)
X_neg, X_test_neg, y_neg, y_test_neg = train_test_split(Data_neg_x, Data_neg_y, test_size= 831, shuffle = False)


# X_train_pos, X_val_pos, y_train_pos, y_val_pos = train_test_split(X_pos, y_pos, test_size= 0.1, shuffle = False)
# X_train_neg, X_val_neg, y_train_neg, y_val_neg = train_test_split(X_neg, y_neg, test_size= 0.1, shuffle = False)



Total_X_train = X_pos + X_neg
Total_y_train = y_pos + y_neg
Total_y_train = np.array(Total_y_train)

Total_X_test = X_test_pos + X_test_neg
Total_y_test = y_test_pos + y_test_neg
Total_y_test = np.array(Total_y_test)
#%%

##-----------------PREPROCESSING------------------------------------


lemma = WordNetLemmatizer()

def preprocessing(data):
    corpus = []
    final_corpus = []
    for i in range(len(data)):
        temp = data[i].lower()
        temp = temp.split()
        x = []
        for j in temp:
            p = contractions.fix(j)
            p = p.split()
            x = x + p

        word = [lemma.lemmatize(k).lower() for k in x]
        sent =  ' '.join(word)
        sent = re.sub('[^a-zA-z0-9]',' ',sent)
        corpus.append(sent)
    
    for q in range(len(data)):
        final_corpus.append(corpus[q].split())
        
    return final_corpus


Total_X_train_p = preprocessing(Total_X_train)
Total_X_test_p = preprocessing(Total_X_test)

max_len = max([len(x) for x in Total_X_train_p])
max_len_test = max([len(x) for x in Total_X_test_p])

#%%
# =============================================================================
# # ------------------------------------- Word2Vec ----------------------------
# =============================================================================

model_w2v = gensim.models.KeyedVectors.load_word2vec_format('D:/IISc/IISc 3rd Semester/DLNLP/Assignments/Assignment 3/Data/GoogleNews-vectors-negative300', binary=True)


#%%
#-------------------------X_train ---------------------------------

dim = 300

X_train = []

for sent in Total_X_train_p:
    temp = []
    count = 0
    for word in sent:
        if word in model_w2v:
            word_embed = list(model_w2v[word])
            temp.append(word_embed)
            count+=1
        

    for i in range(count, max_len):
        zer = [0 for j in range(300)]
        temp.append(zer)
    
    X_train.append(temp)
    
X_train = np.array(X_train) 
print(X_train.shape)

np.save('X_train_embed', X_train)
#%%

# --------------------------- X_test ------------------------------------

dim = 300

X_test = []

for sent in Total_X_test_p:
    temp = []
    count = 0
    for word in sent:
        if word in model_w2v:
            word_embed = list(model_w2v[word])
            temp.append(word_embed)
            count+=1
        

    for i in range(count, max_len_test):
        zer = [0 for j in range(dim)]
        temp.append(zer)
    
    X_test.append(temp)
    
X_test = np.array(X_test) 
print(X_test.shape)

np.save('X_test_embed', X_test)

#%%
del X_train
del X_test

#%%
X_train = np.load('X_train_embed.npy')
X_test = np.load('X_test_embed.npy')

#%%
from keras.layers import LSTM, Activation, Dropout, Dense, Input
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.callbacks import History
import keras
import tensorflow

def LSTM_model():
    model= Sequential()
    model.add(LSTM(64,return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(32,return_sequences = True))
    model.add(LSTM(8))
    model.add(Dense(1, activation = 'sigmoid'))
    
    
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model
 
#%%
print("------------------ Word2Vec Model ---------------------")

model = LSTM_model()
model.summary()
model_word2vec = model.fit(X_train, Total_y_train, epochs = 10, validation_split=0.2)

#%%
print(" ---------- Test Accuracy ------------ ")

test =  model.predict(X_test)

y_predict = []
for i in test:
    if i[0]<0.5:
        y_predict.append(0)
    else:
        y_predict.append(1)
c=0

for i in range(len(Total_y_test)):
    
    if y_predict[i] == Total_y_test[i]:
        c=c+1
print("Test accuracy : ", c/len(Total_y_test))

del y_predict

#%%
# =============================================================================
# #--------------------------------FAST TEXT MODEL-----------------------------
# =============================================================================

model_fasttext = gensim.models.KeyedVectors.load_word2vec_format('D:/IISc/IISc 3rd Semester/DLNLP/Assignments/Assignment 3/Data/fasttext-wiki-news-subwords-300', binary=False)

#%%

#-------------------------------------- X - Train------------------------------

X_train_fasttext = []

for sent in Total_X_train_p:
    temp = []
    count = 0
    for word in sent:
        if word in model_fasttext:
            word_embed = list(model_fasttext[word])
            temp.append(word_embed)
            count+=1
        

    for i in range(count, max_len):
        zer = [0 for j in range(300)]
        temp.append(zer)
    
    X_train_fasttext.append(temp)
    
X_train_fasttext = np.array(X_train_fasttext) 
print(X_train_fasttext.shape)

np.save('X_train_embed_fasttext', X_train_fasttext)


#%%

X_test_fasttext = []

for sent in Total_X_test_p:
    temp = []
    count = 0
    for word in sent:
        if word in model_fasttext:
            word_embed = list(model_fasttext[word])
            temp.append(word_embed)
            count+=1
        

    for i in range(count, max_len_test):
        zer = [0 for j in range(dim)]
        temp.append(zer)
    
    X_test_fasttext.append(temp)
    
X_test_fasttext = np.array(X_test_fasttext) 
print(X_test_fasttext.shape)

np.save('X_test_embed_fasttext', X_test_fasttext)


#%%

del X_train_fasttext
del X_test_fasttext

#%%

X_train_fasttext = np.load('X_train_embed_fasttext.npy')
X_test_fasttext = np.load('X_test_embed_fasttext.npy')


#%%
print("---------------FASTTEXT MODEL------------------")
model_fasttext = model.fit(X_train_fasttext, Total_y_train, epochs = 10, validation_split=0.2)


#%%
print(" ---------- Test Accuracy ------------ ")

test =  model.predict(X_test_fasttext)

y_predict = []
for i in test:
    if i[0]<0.5:
        y_predict.append(0)
    else:
        y_predict.append(1)
c=0

for i in range(len(Total_y_test)):
    
    if y_predict[i] == Total_y_test[i]:
        c=c+1
print("Test accuracy : ", c/len(Total_y_test))

del y_predict
#%%
# =============================================================================
# # -------------------------------- GLOVE MODEL ------------------------------
# =============================================================================
import os

f = open(os.path.join('D:/IISc/IISc 3rd Semester/DLNLP/Assignments/Assignment 3/Data', 'glove.6B.300d.txt'), encoding="utf-8")
c=1
i=0

embedding_glove = {} 
for line in f:
    word = ''
    try:
        values = line.split()
        word = values[0]
        i = i+1
        
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_glove[word] = coefs
        
    except:
        print(c,len(line.split()))
    c+=1

#%%
#-------------------------------------- X - Train------------------------------

X_train_glove = []

for sent in Total_X_train_p:
    temp = []
    count = 0
    for word in sent:
        if word in embedding_glove.keys():
            word_embed = list(embedding_glove.get(word))
            temp.append(word_embed)
            count+=1
        

    for i in range(count, max_len):
        zer = [0 for j in range(300)]
        temp.append(zer)
    
    X_train_glove.append(temp)
    
X_train_glove = np.array(X_train_glove) 
print(X_train_glove.shape)

np.save('X_train_embed_glove', X_train_glove)



#%%

X_test_glove = []

for sent in Total_X_test_p:
    temp = []
    count = 0
    for word in sent:
        if word in embedding_glove.keys():
            word_embed = list(embedding_glove.get(word))
            temp.append(word_embed)
            count+=1
        

    for i in range(count, max_len_test):
        zer = [0 for j in range(dim)]
        temp.append(zer)
    
    X_test_glove.append(temp)
    
X_test_glove = np.array(X_test_glove) 
print(X_test_glove.shape)

np.save('X_test_embed_glove', X_test_glove)


#%%

del X_train_glove
del X_test_glove

#%%

X_train_glove= np.load('X_train_embed_glove.npy')
X_test_glove = np.load('X_test_embed_glove.npy')



#%%
print("--------------------GLOVE MODEL-----------------------")
model_glove = model.fit(X_train_glove, Total_y_train, epochs = 10, validation_split=0.2)


#%%
print(" ---------- Test Accuracy ------------ ")

test =  model.predict(X_test_glove)

y_predict = []
for i in test:
    if i[0]<0.5:
        y_predict.append(0)
    else:
        y_predict.append(1)
c=0

for i in range(len(Total_y_test)):
    
    if y_predict[i] == Total_y_test[i]:
        c=c+1
print("Test accuracy : ", c/len(Total_y_test))

del y_predict
#%%


# =============================================================================
# #-----------------------------Total Embedding--------------------------------
# =============================================================================


Total_embedding_train = np.concatenate((X_train, X_train_fasttext, X_train_glove), axis = 2)
Total_embedding_test = np.concatenate((X_test, X_test_fasttext, X_test_glove), axis = 2)

print(Total_embedding_train.shape)
print(Total_embedding_test.shape)

np.save('Total_embedding_train',Total_embedding_train)
np.save('Total_embedding_test', Total_embedding_test)

#%%

del Total_embedding_train
del Total_embedding_test 

#%%

Total_embedding_train = np.load('Total_embedding_train.npy')
Total_embedding_test = np.load('Total_embedding_test.npy')
print(Total_embedding_train.shape)
print(Total_embedding_test.shape)

#%%

print("--------------META EMBEDDING - MODEL-----------------")
model  = LSTM_model()
model_total = model.fit(Total_embedding_train, Total_y_train, epochs = 10, validation_split=0.2)


#%%
print(" ---------- Test Accuracy ------------ ")

test =  model.predict(Total_embedding_test)

y_predict = []
for i in test:
    if i[0]<0.5:
        y_predict.append(0)
    else:
        y_predict.append(1)
c=0

for i in range(len(Total_y_test)):
    
    if y_predict[i] == Total_y_test[i]:
        c=c+1
print("Test accuracy : ", c/len(Total_y_test))

del y_predict
