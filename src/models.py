'''Models for DRP'''
import torch
import numpy as np
from collections import defaultdict
from torch import nn

class CnnDualInputs(nn.Module):
    def __init__(self, hps):
        super().__init__()
        
        #unpack hps
        out_chans = hps['out_channels']
        lin_ns = hps['linear_neurons']
        lin_ns2 = hps['linear_neurons2']
        drop_p = hps['drop_p']
        
        self.omic_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out_chans, kernel_size=16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(out_chans, 8, 16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.LazyLinear(lin_ns),
            #nn.BatchNorm1d(lin_ns),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(lin_ns, lin_ns),
            #nn.BatchNorm1d(lin_ns),
            nn.ReLU(),
        )
        self.con_ff = nn.Sequential(
            nn.LazyLinear(lin_ns2),
            #nn.BatchNorm1d(lin_ns2),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(lin_ns2, lin_ns2), 
            nn.ReLU(), 
            nn.LazyLinear(1))

    def forward(self, xo, xd):
        #x = self.flatten(x)
        omic_encoded = self.omic_encoder(xo)
        concat = torch.cat((omic_encoded, xd), dim=1) #dim=0 gives batchsize
        output = self.con_ff(concat)
        
        return output
    
class MeanModel():
    '''Creates benchmark model that uses mean truth values for prediciton.
    
    For a given drug cell lines pair, d1 c1 in the test set the model predicts
    the mean truth value of all drug cell line pairs in the traning set 
    that include d1.
    
    All drugs need to be in the traning set.
    
    Inputs
    -----
    
    y_train: pd series or dataframe
    traning truth values were index of the df gives cell_line drug 
    seprated by the string :: e.g. for d1 and cl1 index = 'd1::cl1'
    
    all_drugs: list like
    gives all drug names 
    
    Methods
    ------
    predict(y_index): gives models prediciton for cell line drug pairs
    
    replace_nan(re='mean') replace missing values with re
    '''
    
    def __init__(self, y_train, drugs, verb=1):
        model = defaultdict(list)
        #group cls by drugs
        self.verb = verb
        for ind, val in y_train.items():
            cl, d = ind.split('::')
            model[d].append(val)
            
        #take average of all values for a given drug   
        for d in drugs:
            model[d] = np.mean(model[d])
        
        self.model = model
        
    def predict(self, y_index, reformat=True):
        #reformat index to get just drug
        if reformat:
            y_index = [y.split('::')[1] for y in y_index]
            
        return np.array([self.model[y] for y in y_index])
    
    def replace_nan(self, re='mean'):
        #replace nan's with re, deflat re=0
        num_nan = 0
        for k in self.model:
            if np.isnan(self.model[k]):
                num_nan += 1
                if re=='mean':
                    vals = np.array(list(self.model.values()))
                    vals = vals[~np.isnan(vals)]
                    self.model[k] = vals.mean()
                else:
                    self.model[k] = re
        if self.verb > 0:
            print(f'{num_nan} nan values replaced out of {len(self.model)}')
