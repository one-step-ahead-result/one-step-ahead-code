import numpy as np
import torch
import re
import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch import optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class Glove():
    
    def __init__(self, fUrl):
        """
        load model(no-binary model)
        """
        with open(fUrl) as f:
            self.word_dic = {line.split()[0]:torch.tensor(np.asarray(line.split()[1:], dtype='float'),dtype=torch.float32) for line in f}
    
    def isinkey(self,word):
        return word in self.word_dic.keys()
    
    def __getitem__(self,index):
        return self.word_dic[index]

class TextCNN(nn.Module):
    '''
    Binary classify model.
    '''
    def __init__(self,filter_num,filter_sizes,vocabulary_size,embedding_dim,dropout=0.5):
        '''
        filter_num: number of the convolutional kernel.
        filter_sizes: size of the convolutional kernels.
        vocabulary_size: max length of the input news.
        embedding_dim: embedding size.
        '''
        super(TextCNN, self).__init__()
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.vocabulary_size = vocabulary_size
        embedding_dimension =embedding_dim
        self.conv1 = nn.Conv2d(1,3,(1,embedding_dim))
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.att=nn.ModuleList(
            [nn.Linear(3*size+1,1) for size in filter_sizes])
        self.linear=nn.ModuleList(
            [nn.Linear(vocabulary_size-size+1,1) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, 1)

    def forward(self, x, y=None):
        conv1 = self.conv1(x)
        conv1=[torch.cat([torch.cat([(conv1[:,:,i:l+i,:]).reshape(-1,1,1,3*l) for _ in range(self.filter_num)],1) for i in range(self.vocabulary_size-l+1)],2) for l in self.filter_sizes]
        x = [F.relu(conv(x)) for conv in self.convs]
        atten = [torch.cat([item,c],3) for item,c in zip(x,conv1)]
        atten = [F.sigmoid(att(item)).squeeze(3) for item,att in zip(atten,self.att)]
        x = [F.relu(item.squeeze(3)*att) for item,att in zip(x,atten)]
        x = [F.relu(l(item)) for item,l in zip(x,self.linear)]
        x = torch.cat(x, 1).squeeze(2)
        x = self.dropout(x)
        if y is None:
            return F.sigmoid(self.fc(x))
            return F.softmax(self.fc(x))#.argmax(1)
        else:
            return F.binary_cross_entropy(F.sigmoid(self.fc(x)),y)
            return F.l1_loss(F.softmax(self.fc(x)),y)
            
def t2v(text):
    # Vectorize the text.
    re=[]
    for i in tokener(text.lower()):
        if w2v.isinkey(i):
            re.append(w2v[i])
    while len(re)<20:
        re.append(torch.zeros(50,dtype=torch.float32))
    return torch.cat(re,0).reshape(1,-1,50)[:,:20,:]

# load the word2vec pre-trained model.
w2v=Glove('glove.6B.50d.txt')

# load train data.
df_incident=pd.read_csv('incident_label.csv',sep='\t')
df_NYT=pd.read_csv("new_NYT.csv",sep='\t')
tokener=nltk.tokenize.word_tokenize
df_NYT['token']=df_NYT['title'].map(lambda x:tokener(x))
df_NYT['length']=df_NYT['token'].map(lambda x:len(x))
NYT_train=df_NYT.loc[df_NYT['length']>5].loc[df_NYT['length']<=20]
NYT_train['vec']=NYT_train['title'].map(lambda x:t2v(x))
df_incident['token']=df_NYT['title'].map(lambda x:tokener(x))
df_incident['length']=df_incident['token'].map(lambda x:len(x))
incident_train=df_incident.loc[df_incident['length']>5].loc[df_incident['length']<=20]
incident_train['vec']=incident_train['title'].map(lambda x:t2v(x))

X=[]
Y=[]
for i in incident_train['vec']:
    X.append(i)
    Y.append(1)
for i in NYT_train['vec']:
    X.append(i)
    Y.append(0)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=0)

X_train=torch.cat(x_train)
Y_train=torch.tensor(y_train,dtype=torch.float32)
X_eval=torch.cat(x_test)
Y_eval=torch.tensor(y_test,dtype=torch.float32)

traindata=TensorDataset(X_train,Y_train)
evaldata=TensorDataset(X_eval,Y_eval)

model=TextCNN(2,[2,3,4],20,50).cuda(device=0)
optimizer = optim.Adam(model.parameters(),lr=0.01)
dataloader=DataLoader(traindata,batch_size=40000)

model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(),lr=0.01)

# Train model.
for i in range(200):
    model=model.train()
    for x,y in dataloader:
        x=x.reshape(-1,1,20,50).cuda()
        y=y.reshape(-1).cuda()
        loss=model(x,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if i%20==0:
        model=model.eval()
        print(classification_report(Y_eval.tolist(),[[int(i[0]>0.5)] for i in model(X_eval.reshape(-1,1,20,50).cuda()).tolist()],digits=4))

# Predict the reuters news.
df_reuter=pd.read_csv('reuter.csv',sep='\t')
df_reuter['token']=df_reuter['title'].map(lambda x:tokener(x))
df_reuter['vec']=df_reuter['title'].map(lambda x:t2v(x))
X_test=torch.cat(df_reuter['vec'].tolist())
X_test=TensorDataset(X_test)
X_test=DataLoader(X_test,batch_size=20000)
Y=[]
for i in X_test:
    x=i[0].reshape(-1,1,20,50).cuda()
    y=model(x)
    Y.append(y)
Y=torch.cat(Y).reshape(-1)
df_reuter['Y']=Y.tolist()
df_reuter=df_reuter.loc[df_reuter['Y']==1].reset_index()
df_reuter.to_csv('reuter_incident.csv',sep='\t')