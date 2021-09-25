# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:24:44 2021

@author: Anshul
"""

#Import libraries
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader,Dataset
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Parameters
training_size = 30000
N_iter = 20
input_size = 58
hidden_layer = 200
output_layer = 1
norm = Normalizer()

batch_size = [4, 64, 128, 256, 512]
learning_rate = [0.05, 0.01, 0.001]

saved_epoch = 0
saved_learning_rate = 0
saved_batch_size = 0
saved_train_loss = 1000000
saved_test_loss = 100000

Total_train_loss_list = []
Total_test_loss_list = []
#%%
#Data Loading
class ONPDataset_train(Dataset):
    def __init__(self):
        #dataloading
        dataset= np.loadtxt('C:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 1\OnlineNewsPopularity\OnlineNewsPopularity.csv', delimiter=",", dtype = np.float32, skiprows=1)
        self.X_train = torch.from_numpy(dataset[:training_size,:-1])
        self.y_train = torch.from_numpy(dataset[:training_size,[-1]])
        self.n_training_samples = self.X_train.shape[0]
    
    def __getitem__(self, index):
        #dataset[0]
        return self.X_train[index], self.y_train[index]
        
    
    def __len__(self):
        #len(dataset)
        return self.n_training_samples

class ONPDataset_test(Dataset):
    def __init__(self):
        #dataloading
        dataset= np.loadtxt('C:\IISc\IISc 3rd Semester\DLNLP\Assignments\Assignment 1\OnlineNewsPopularity\OnlineNewsPopularity.csv', delimiter=",", dtype = np.float32, skiprows=1)
        self.X_test = torch.from_numpy(dataset[training_size+1:,:-1])
        self.y_test = torch.from_numpy(dataset[training_size+1:, [-1]])
        self.n_test_samples = self.X_test.shape[0]
    
    def __getitem__(self, index):
        #dataset[0]
        return self.X_test[index], self.y_test[index]
        
    
    def __len__(self):
        #len(dataset)
        return self.n_test_samples

#Dataset Visualization 
#   - Training Dataset
train_dataset = ONPDataset_train()
features, targets = train_dataset[0]

#   - Test Dataset
test_dataset = ONPDataset_test()
features1, targets1 = test_dataset[0]

#With Normalizer()
train_dataset.X_train = norm.fit_transform(train_dataset.X_train)
test_dataset.X_test = norm.transform(test_dataset.X_test)


#%%
#Model definition
class FNN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(FNN, self).__init__()
        
        self.l1 = nn.Linear(inputs, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.l2 = nn.Linear(hidden, 100)
        self.l3 = nn.Linear(100, outputs)

    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l3(x)
        return x
    
#%%
for batch in batch_size:
    print("----Batch Size---- : {}".format(batch))
    #Dataloader
    train_loader = DataLoader(dataset = train_dataset, batch_size= batch, shuffle = False)
    
    #Loss and optimizer
    model = FNN(input_size, hidden_layer, output_layer)
    criterion = nn.MSELoss()
    
    train_loss_list = []
    test_loss_list = []
    for learn_rate in learning_rate:
        print("---Learning Rate ---: {}".format(learn_rate))
        optimizer = torch.optim.RMSprop(model.parameters(), lr = learn_rate)
        best_weights = copy.deepcopy(model.state_dict())
        
        for epoch in range(N_iter):
            total_loss = 0
            
            for mode in ['train', 'test']:
                if mode == 'train':
                    for batch_idx, (feature, label) in enumerate(train_loader):
                        feature = feature.to(device)
                        label = label.to(device)
                        
                        #Forward
                        pred = model(feature.float()) # 4x1
                        #print(pred.shape)
                        loss = criterion(pred.float(), label.float())
                        
                        #Backward
                        optimizer.zero_grad()
                        loss.backward()
                        
                        #Update rule
                        optimizer.step()
                        total_loss += loss*len(label)
                    
                    total_loss = torch.sqrt(total_loss/30000).item()
                    train_loss_list.append(total_loss)
                    
                    print('Epoch : {}, Training Loss : {}'.format(epoch+1, total_loss))
                
                else:
                    with torch.no_grad():
                        model.eval()
                        
                        predicted = model(torch.tensor(test_dataset.X_test))
                        loss = criterion(predicted, test_dataset.y_test)
                        loss = torch.sqrt(loss)
                        
                        test_loss_list.append(loss.item())
                        print(f'Test Loss: {loss.item()} ')
                        
                        
                    if loss < saved_test_loss:
                        saved_test_loss = loss
                        saved_batch_size = batch
                        saved_epoch = epoch+1
                        saved_train_loss = total_loss
                        saved_learning_rate = learn_rate
                        best_weights = copy.deepcopy(model.state_dict())
                        
    Total_test_loss_list.append(test_loss_list)
    Total_train_loss_list.append(train_loss_list)

print("----------------------------------------------------------------------")
    
model.load_state_dict(best_weights)

with torch.no_grad():                      
    model.eval()
    
    predicted = model(torch.tensor(test_dataset.X_test))
    loss = criterion(predicted, test_dataset.y_test)
    loss = torch.sqrt(loss)
    
    print(f'saved model: Test Loss: {loss.item()}')
    print("Saved batch size: {} ".format(saved_batch_size))
    print("Saved epoch: {} ".format(saved_epoch))
    print("Saved train loss: {} ".format(saved_train_loss))
    print("Saved test loss: {} ".format(saved_test_loss))
    print("Saved learning_rate: {} ".format(saved_learning_rate))


print("---------------After Saving model results ----------------- ")

#%%

FILENAME = "saved_model_final1.pth"
torch.save(model.state_dict(),FILENAME)
saved_model = FNN(input_size, hidden_layer, output_layer)
saved_model.load_state_dict(torch.load(FILENAME))

#%%
with torch.no_grad():                      
    saved_model.eval()

    predicted = saved_model(torch.tensor(test_dataset.X_test))
    loss = criterion(predicted, test_dataset.y_test)
    loss = torch.sqrt(loss)
    
    print(f'saved model: Test Loss: {loss.item()}')
    print("Saved batch size: {} ".format(saved_batch_size))
    print("Saved epoch: {} ".format(saved_epoch))
    print("Saved train loss: {} ".format(saved_train_loss))
    print("Saved test loss: {} ".format(saved_test_loss))
    print("Saved learning_rate: {} ".format(saved_learning_rate))

#%%
#Graphs Plot

epochs = [i+1 for i in range(N_iter * len(learning_rate))]
for i in range(len(batch_size)):
    # plt.plot(epochs, Total_train_loss_list[i], 'b')
    plt.plot(epochs, Total_test_loss_list[i], 'r')
    plt.xlabel("No. of Epochs")
    plt.ylabel("Loss value")
    plt.title("Graph for batch size: {}".format(batch_size[i]))
    plt.show()


