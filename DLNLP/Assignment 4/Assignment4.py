# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 18:29:54 2021

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
import pandas as pd
from nltk.corpus import stopwords

#%%

DATA_PATH = 'D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\TrainData.csv'
Data  = pd.read_csv(DATA_PATH , sep = ',')

X_data = Data['Text']
y_data = pd.get_dummies(Data['Category'])

Data["Category"] = Data["Category"].astype('category').cat.codes
y_cat_data = Data["Category"]


#%%

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

        word = [lemma.lemmatize(k).lower() for k in x if k not in set(stopwords.words('english'))]
        sent =  ' '.join(word)
        sent = re.sub('[^a-zA-z]',' ',sent)
        corpus.append(sent)
    
    for q in range(len(data)):
        final_corpus.append(corpus[q].split())
        
    return final_corpus

X_data_p = preprocessing(X_data)


#%%
# =============================================================================
# # ------------------------------------- Word2Vec ----------------------------
# =============================================================================

model_w2v = gensim.models.KeyedVectors.load_word2vec_format('D:/IISc/IISc 3rd Semester/DLNLP/Assignments/Assignment 3/Data/GoogleNews-vectors-negative300', binary=True)


#%%

#-------------------------X_train embedding with max-len = 1670 ---------------------------------

dim = 300
max_len = max([len(x) for x in X_data_p])

X_train = []

for sent in X_data_p:
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
    
X_train_1670 = np.array(X_train) 
print(X_train_1670.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_1670', X_train_1670)

#%%
del X_train 

#%%
del X_train_1670

#%%
X_train_1670 = np.load('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_1670.npy')
X_train_dim1 = X_train_1670.shape[0]
X_train_dim2 = X_train_1670.shape[1]
X_train_dim3 = X_train_1670.shape[2]

#%%
#X_train_1670 = np.load('X_train_embed.npy')

#%%
#-------------------------X_train embedding with max-len = 500 ---------------------------------

dim = 300
max_len = 500
X_train = []

for sent in X_data_p:
    temp = []
    count = 0
    for word in sent:
        if count<max_len:
            if word in model_w2v:
                word_embed = list(model_w2v[word])
                temp.append(word_embed)
                count+=1    

    for i in range(count, max_len):
        zer = [0 for j in range(dim)]
        temp.append(zer)
    
    X_train.append(temp)
    
X_train_500 = np.array(X_train) 
print(X_train_500.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_500', X_train_500)

#%%
del X_train

#%%
del X_train_500

#%%
X_train_500 = np.load('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_500.npy')
X_train_dim1 = X_train_500.shape[0]
X_train_dim2 = X_train_500.shape[1]
X_train_dim3 = X_train_500.shape[2]


#%%
#-------------------------X_train embedding with max-len = 800 ---------------------------------

dim = 300
max_len = 800
X_train = []

for sent in X_data_p:
    temp = []
    count = 0
    for word in sent:
        if count<max_len:
            if word in model_w2v:
                word_embed = list(model_w2v[word])
                temp.append(word_embed)
                count+=1    

    for i in range(count, max_len):
        zer = [0 for j in range(dim)]
        temp.append(zer)
    
    X_train.append(temp)
    
X_train_800 = np.array(X_train) 
print(X_train_800.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_500', X_train_800)

#%%
del X_train

#%%
del X_train_800

#%%
X_train_800 = np.load('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_500.npy')
X_train_dim1 = X_train_800.shape[0]
X_train_dim2 = X_train_800.shape[1]
X_train_dim3 = X_train_800.shape[2]

#%%
#-------------------------X_train embedding with max-len = 1000 ---------------------------------

dim = 300
max_len = 1000
X_train = []

for sent in X_data_p:
    temp = []
    count = 0
    for word in sent:
        if count<max_len:
            if word in model_w2v:
                word_embed = list(model_w2v[word])
                temp.append(word_embed)
                count+=1    

    for i in range(count, max_len):
        zer = [0 for j in range(dim)]
        temp.append(zer)
    
    X_train.append(temp)
    
X_train_1000 = np.array(X_train) 
print(X_train_1000.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_1000', X_train_1000)

#%%
del X_train
del X_train_1000

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
#-------------------------------------- X - Train embedding - 1670 ------------------------------

X_train_glove = []
max_len = max([len(x) for x in X_data_p])

for sent in X_data_p:
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
    
X_train_glove_1670 = np.array(X_train_glove) 
print(X_train_glove_1670.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_glove_1670', X_train_glove_1670)
#%%

del X_train_glove
del X_train_glove_1670

#%%
#-------------------------------------- X - Train embedding - 1000 ------------------------------

X_train_glove = []
max_len = 1000

for sent in X_data_p:
    temp = []
    count = 0
    for word in sent:
        if count<max_len:
            if word in embedding_glove.keys():
                word_embed = list(embedding_glove.get(word))
                temp.append(word_embed)
                count+=1

    for i in range(count, max_len):
        zer = [0 for j in range(300)]
        temp.append(zer)
    
    X_train_glove.append(temp)
    
X_train_glove_1000 = np.array(X_train_glove) 
print(X_train_glove_1000.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_glove_1000', X_train_glove_1000)

#%%

del X_train_glove

#%%
del X_train_glove_1000

#%%
X_train_glove_1000 = np.load('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_glove_1000.npy')
#%%
X_train_dim1 = X_train_glove_1000.shape[0]
X_train_dim2 = X_train_glove_1000.shape[1]
X_train_dim3 = X_train_glove_1000.shape[2]

#%%
#---------------------------- X - Train embedding - 800 -----------------------

X_train_glove = []
max_len = 800

for sent in X_data_p:
    temp = []
    count = 0
    for word in sent:
        if count<max_len:
            if word in embedding_glove.keys():
                word_embed = list(embedding_glove.get(word))
                temp.append(word_embed)
                count+=1

    for i in range(count, max_len):
        zer = [0 for j in range(300)]
        temp.append(zer)
    
    X_train_glove.append(temp)
    
X_train_glove_800 = np.array(X_train_glove) 
print(X_train_glove_800.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_glove_800', X_train_glove_800)

#%%

del X_train_glove

#%%
del X_train_glove_800

#%%
X_train_glove_800 = np.load('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_glove_800.npy')
X_train_dim1 = X_train_glove_800.shape[0]
X_train_dim2 = X_train_glove_800.shape[1]
X_train_dim3 = X_train_glove_800.shape[2]


#%%
#-------------------------------------- X - Train embedding - 500 ------------------------------

X_train_glove = []
max_len = 500

for sent in X_data_p:
    temp = []
    count = 0
    for word in sent:
        if count<max_len:
            if word in embedding_glove.keys():
                word_embed = list(embedding_glove.get(word))
                temp.append(word_embed)
                count+=1

    for i in range(count, max_len):
        zer = [0 for j in range(300)]
        temp.append(zer)
    
    X_train_glove.append(temp)
    
X_train_glove_500 = np.array(X_train_glove) 
print(X_train_glove_500.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_glove_500', X_train_glove_500)

#%%

del X_train_glove

#%%
del X_train_glove_500

#%%
X_train_glove_500 = np.load('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_train_embed_glove_500.npy')
X_train_dim1 = X_train_glove_500.shape[0]
X_train_dim2 = X_train_glove_500.shape[1]
X_train_dim3 = X_train_glove_500.shape[2]

#%%

# =============================================================================
# # -----------------------C-LSTM Model ---------------------------------------
# =============================================================================


from keras.layers import LSTM, Conv1D, Activation, Dropout, Dense, Input, MaxPooling1D, Attention
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.callbacks import History
import keras
import tensorflow
from keras_self_attention import SeqSelfAttention
#%%
import matplotlib.pyplot as plt

def plot_graph(model, epochs, string):
    Train_loss =  model.history.get('loss')
    Val_loss    =  model.history.get('val_loss')
    Train_accuracy  =  model.history.get('accuracy')
    Val_accuracy    =  model.history.get('val_accuracy')
    Epochs = [i for i in range(epochs)]
    
    plt.plot(Epochs, Train_loss, label = 'Train loss')
    plt.plot(Epochs, Val_loss, label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Plot for {}".format(string))
    plt.show()
    
    
    plt.plot(Epochs, Train_accuracy, label = 'Train accuracy')
    plt.plot(Epochs, Val_accuracy, label = 'Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title("Accuracy Plot for {}".format(string))
    plt.legend()
    plt.show()

#%%
def C_LSTM_Model():
    model = Sequential()
    model.add(Conv1D(filters=100, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(LSTM(100, dropout = 0.2, recurrent_dropout= 0.1, return_sequences=True))
    model.add(LSTM(50))
    model.add(SeqSelfAttention(attention_activation='softmax', name='Attention'))
    model.add(Dense(5, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.build((X_train_dim1, X_train_dim2, X_train_dim3))
    
    return model
    
#%%
## --------------------- Word2Vec - 500 Model Train ---------------------------
epoch = 32
model = C_LSTM_Model()
model.summary()
model_w2v_500 = model.fit(X_train_500, y_cat_data, batch_size= 16, epochs = epoch, validation_split=0.1)

#%%

#--------------- Plot Graph for Word2Vec - 500 Model Train --------------------
plot_graph(model_w2v_500, epoch, 'W2V_500 model')


#%%
model.save("D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_model\W2V_Model_500_50")

#%%
# =============================================================================
# ## ------------------ Glove - 500 Model Train -------------------------------
# =============================================================================

epoch = 20
model = C_LSTM_Model()
model.summary()
model_glove_500 = model.fit(X_train_glove_500, y_cat_data, batch_size= 16, epochs = epoch, validation_split=0.2)

#%%
plot_graph(model_glove_500, epoch, 'Glove_500 model')

#%%
model.save("D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_model\Glove_Model_New_500_20")

#%%
# =============================================================================
# ## ------------------ Glove - 800 Model Train -------------------------------
# =============================================================================

epoch = 30
model = C_LSTM_Model()
model.summary()
model_glove_800 = model.fit(X_train_glove_800, y_cat_data, batch_size= 16, epochs = epoch, validation_split=0.2)

#%%
plot_graph(model_glove_800, epoch, 'Glove_800 model')

#%%
model.save("D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_model\Glove_Model_800_35")

#%%

# =============================================================================
# # ------------------------------- TEST DATA W2V ---------------------------------
# =============================================================================

TEST_DATA_PATH = 'D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\TestData_Inputs.csv'
Test_Data  = pd.read_csv(TEST_DATA_PATH , sep = ',')

X_test = Test_Data['Text']
#y_test = pd.get_dummies(Test_Data['Category'])

#y_cat_test = Test_Data["Category"].astype('category').cat.codes

#%%

#Test - Preprocessing 
X_test_p = preprocessing(X_test)

#%%

dim = 300
max_len_test = 500
X_test = []
for sent in X_test_p:
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
    
X_test_w2v_500 = np.array(X_test) 
print(X_test_w2v_500.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_test_w2v_500', X_test_w2v_500)

#%%

# =============================================================================
# # ------------------------------- TEST DATA GLove----------------------------
# =============================================================================


X_test_glove = []
dim = 300
max_len_test = 500

for sent in X_test_p:
    temp = []
    count = 0
    for word in sent:
        if count<max_len_test:
            if word in embedding_glove.keys():
                word_embed = list(embedding_glove.get(word))
                temp.append(word_embed)
                count+=1
        

    for i in range(count, max_len_test):
        zer = [0 for j in range(dim)]
        temp.append(zer)
    
    X_test_glove.append(temp)
    
X_test_glove_500 = np.array(X_test_glove) 
print(X_test_glove_500.shape)

np.save('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_test_glove_500', X_test_glove_500)

#%%

del X_test_glove

#%%

del X_test_glove_500

#%%

X_test_glove_500 = np.load('D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\saved_variables\X_test_glove_500.npy')

#%%

Prediction = model.predict(X_test_glove_500)
print(Prediction.shape)

#%%
# =============================================================================
# #---------------------------Test Labels -------------------------------------
# =============================================================================

TEST_DATA_PATH = 'D:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 4\Assignment4_TestLabels.csv'
Test_Labels  = pd.read_csv(TEST_DATA_PATH , sep = ',')

#y_test = Test_Labels['Labels'].astype('category').cat.codes
y_test = Test_Labels['Category'].astype('category').cat.codes
#%%

pred = np.argmax(Prediction, axis = 1)

#%%

from sklearn import metrics
fscore = metrics.f1_score(y_test, pred, average = 'micro')
print(fscore)
#%%

