''' Training loop functions for pytorch'''

import torch
from torch import nn
import numpy as np
from scipy.stats import pearsonr
device = "cuda" if torch.cuda.is_available() else "cpu"
#early stopping implementation
class EarlyStopping:
    '''Class to implement early stopping
    
     ------inputs------
    model: PyTorch model 
    
    patience: int, defalt=10
    number of epochs the model can not meet the improvement criteria before
    early stopping triggers
    
    delta: int, defalt=0
    How much the loss needs to decrease by to count as an improvement
    e.g. delta=1 means the loss needs to be at least 1 less than previous best loss
    '''
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.count = 0
        self.min_val_loss = np.inf
        self.best_model_dict = None
        self.best_epoch = None
        
    def earily_stop(self, val_loss, model_state, e=0):
        #if loss improves 
        if val_loss < self.min_val_loss:
            print(f'loss improved from {self.min_val_loss} to {val_loss}')
            self.min_val_loss = val_loss
            self.count = 0 
            #self.best_model_dict = model_state
            #Save
            torch.save(model_state, 'early_stop_temp')
        #if loss does not improved more than delta 
        elif val_loss >= (self.min_val_loss + self.delta):
            self.count += 1
            if self.count >= self.patience: 
                return True
            
        return False #if stopping contions not met
        


#train loop for mutiple dataloaders 
def tl_multi_dls(train_dls=None, y_train=None, val_dls=None, y_val=None, 
                 model=None, optimiser=None, loss_fn=nn.MSELoss(), 
                 epochs=10, verb=1, early_stopping_dict=None):
    '''torch train loop for mutiple data loaders
    
    ------inputs------
    train_dls: list like 
    contains multiple inputs where each input is a data loader 
    for the traning data
    
    y_train: DataLoader
    dataloader with the target traning values
    
    val_dls: list like
    contains multiple inputs where each input is a data loader
    for the validaiton data
    
    y_val: DataLoader
    dataloader with the target valdiation values
    
    
    early_stopping_dict: dict
    contains earily stopping params, patience and delta 
    defalt patience=10 and delta=0.0
    
    ------returns------
    train_hist, best_model_dict
    
    if early stopping implemented best model is used to overwrite model
    this happens even when early stopping is't tirggered
    '''
    train_hist = {'train_loss': [], 'val_loss': []}
    best_model_dict = None
    assert type(train_dls) == list
    assert train_dls and y_train and val_dls and y_val != None
    assert model and optimiser != None
    
    #early stopping
    if early_stopping_dict and val_dls:
        p, d = early_stopping_dict['patience'], early_stopping_dict['delta']
        early_stopper = EarlyStopping(patience=p, delta=d)
    
    for e in range(epochs):
        #train_dls_y = train_dls
        #train_dls_y.append(y_train)
        loss_train = 0.0
        model.train()
        for batch, (*x, y) in enumerate(zip(*train_dls, y_train)):
            x = [inp.to(device) for inp in x]
            y = y.to(device)

            # Compute prediction error
            pred = model(*x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            loss_train += loss.item()
            
        train_hist['train_loss'].append(loss_train / len(train_dls[0]))
        
        if val_dls:
            #val_dls_y = val_dls
            #val_dls_y.append(y_val)
            #find validaiton loss
            val_loss = 0.0
            r2 = 0.0
            mse_val = 0.0 

            model.eval()   
            with torch.no_grad():
                for *xv, yv in zip(*val_dls, y_val):
                    xv = [inp.to(device) for inp in xv]
                    yv = yv.to(device)
                    val_pred = model(*xv)
                    #val_pred = val_pred.reshape(len(val_pred))
                    v_loss = loss_fn(val_pred, yv)
                    val_loss += v_loss.item()
                    yv = yv.cpu().detach().numpy()
                    val_pred = val_pred.cpu().detach().numpy()
            train_hist['val_loss'].append(val_loss / len(val_dls[0]))
            
            #eairly stopping
            if early_stopping_dict:
                if early_stopper.earily_stop(val_loss, model.state_dict()):
                    print('')
                    print(f'stopping early at epoch {e + 1}')
                    print(f'best epoch {e + 1 - early_stopper.patience}')
                    #best_model_dict  = early_stopper.best_model_dict
                    #model.load_state_dict(best_model_dict)
                    #model.eval()
                    break
                    
        if verb > 0:
            print(f'Epoch {e + 1}\n-----------------')
            ftl = loss_train / len(train_dls[0])
            if val_dls:
                fvl = val_loss / len(val_dls[0])
                print(f'Train loss: {ftl:>5f}, Val loss: {fvl:>5f}')
                print(pearsonr(yv.reshape(len(yv)), 
                               val_pred.reshape(len(val_pred))))
            else:
                print(f'Train loss: {ftl:>5f}')
                
     #load best model if early stoppng triggers or not
    if early_stopping_dict:
        print('loading best model')
        #best_model_dict  = early_stopper.best_model_dict
        #model.load_state_dict(best_model_dict)
        model.load_state_dict(torch.load('early_stop_temp'))
        model.eval()
        
    return train_hist