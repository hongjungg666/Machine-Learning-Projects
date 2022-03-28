import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


# if a word is not in your vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc):
        self.df = df_enc

    def __len__(self):
        """ Length of the dataset """
        ### BEGIN SOLUTION
        L = self.df.shape[0]-4
        ### END SOLUTION
        return L

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        
        x[idx] should be a numpy array of shape (5,)
        """
        ### BEGIN SOLUTION
        x = np.array(self.df.iloc[idx:idx+5,0])
        y = self.df.iloc[idx+2,1]
        
        ### END SOLUTION
        return x, y


def label_encoding(cat_arr):
    """ 
      Given a numpy array of strings returns a dictionary with label encodings.

   First take the array of unique values and sort them (as strings). 
   """
    ### BEGIN SOLUTION
    uniq = np.sort(np.unique(cat_arr.astype('str')))
    vocab2index = {o:i for i,o in enumerate(uniq)}
    ### END SOLUTION
    return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column

    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()
    ### BEGIN SOLUTION
    df_enc["word"]  = [vocab2index[x] if x in vocab2index.keys() else V for x in df_enc['word']]
    df_enc['label'] = df_enc['label'].map(label2index)
    ### END SOLUTION
    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)
        ### BEGIN SOLUTION
        self.vocab_embed = nn.Embedding(vocab_size,emb_size)
        self.linear = nn.Linear(5*emb_size, n_class)# prepare for the cross_entropy func K*embed
        self.flatten = nn.Flatten()
        ### END SOLUTION
        
    def forward(self, x):
        """Apply the model to x
        
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size 
           .flatten works
        3. Apply a linear layer
        """
        ### BEGIN SOLUTION
        x_emb = self.vocab_embed(x)
#         x_emb = torch.flatten(x_emb, s tart_dim=1) #or x.flatten(1) axis
        x_emb = self.flatten(x_emb)
        x = self.linear(x_emb)
        ### END SOLUTION
        return x

def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    for i in range(epochs):
        ### BEGIN SOLUTION
        for x, y in train_dl:
            model.train()
            x= x.type(torch.LongTensor)
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)# input (Tensor) – (N, C)(N,C) where C = number of classes
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = loss.item()
        ### END SOLUTION
        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print("train loss  %.3f val loss %.3f and accuracy %.3f" % (
            train_loss, valid_loss, valid_acc))

def valid_metrics(model, valid_dl):
    ### BEGIN SOLUTION
    model.eval()
    y_lst = []
    y_pred_lst = []
    for x,y in valid_dl:
        x= x.type(torch.LongTensor)
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
#         sof = y_hat.softmax(dim=1)# get prediction
#         y_pred = torch.max(sof, 1)[1] # get actual class
        y_lst.append(y.detach().numpy())
        y_pred_lst.append(y_hat.detach().numpy())
    y_preds = np.concatenate(y_pred_lst, axis=0)
    ys = np.concatenate(y_lst, axis=0)
    loss = F.cross_entropy(torch.FloatTensor(y_preds), torch.tensor(ys))
    y_pred = np.argmax(y_preds, axis=1)
    
    accuracy = np.sum(y_pred==ys)/y_pred.shape[0]
    val_loss = loss.item()
    val_acc = accuracy.item()
    ### END SOLUTION
    return val_loss, val_acc

