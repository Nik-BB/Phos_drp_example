import sklearn
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_all_drugs(x, xd, y):
    '''Create data for all drug and cell line pairs, for use in models.
    
    With cell line data (x) that is not drug spesfic (i.e. the same for 
    all drugs) copies this data for each drug while removing missing values 
    that are contained in y as nan.
    The indexes in the dataframes created agree with each other. 
    E.g. the zeorth index of the dataframes corrisponds to the 
    drug cell line pair given by x.iloc[0], y.iloc[0].
    
    Inputs
    -------
    x: pd dataframe.
    Omic data (i.e. phospo) where the index is the cell lines
    and cols are features.
    
    xd: pd dataframe.
    One hot encoded representation of the drugs.
    
    y: pd datafame.
    Target values (i.e. ic50 values) where the index is 
    the cell lines and cols are the drugs. 
    
    Returns
    -------
    x_final: pd dataframe.
    Omics data for all drugs and cell lines
    
    X_drug_final: pd dataframe.
    One hot enocding for all drugs and cell lines
    
    y_final: pd index
    Target values for all drugs and cell lines

    '''
    drug_inds = []
    x_dfs = []
    x_drug_dfs = []
    y_final = []
    
    x.astype(np.float16)
    for i, d in enumerate(xd.columns):
        #find cell lines without missing truth values
        y_temp = y[d]
        nona_cells = y_temp.index[~np.isnan(y_temp)]
        #finds the index for the start / end of each drug
        ind_high = len(nona_cells) + i
        drug_inds.append((d, i, ind_high))
        i += len(nona_cells)

        #store vals of the cell lines with truth values
        x_pp = x.loc[nona_cells] 
        x_dfs.append(x_pp)
        X_drug = pd.DataFrame([xd[d]] * len(x_pp))
        x_drug_dfs.append(X_drug)
        y_final.append(y_temp.dropna())

    #combine values for all drugs  
    x_final = pd.concat(x_dfs, axis=0)
    X_drug_final = pd.concat(x_drug_dfs, axis=0)
    y_final = pd.concat(y_final, axis=0)
    
    #reformat indexs 
    cls_drugs_index = x_final.index + '::' + X_drug_final.index 
    x_final.index = cls_drugs_index
    X_drug_final.index = cls_drugs_index
    y_final.index = cls_drugs_index
    
    x_final.astype(np.float32)
    X_drug_final.astype(np.float32)
    y_final.astype(np.float32)
    
    return x_final, X_drug_final, y_final


def into_dls(x: list, batch_size=512):
    '''helper func to put DRP data into dataloaders
    for non gnn x[0], x[1] and x[2] give the omics, drug and target values
    respectively. for gnn x[0] gives graph data
    
    '''
    #checks 
    assert len(x[0]) == len(x[1])
    assert len(x[0]) == len(x[2])
    from torch_geometric.loader import DataLoader as DataLoaderGeo 
    from torch.utils.data import DataLoader
    import torch_geometric.data as tgd
    
    x[1] = DataLoader(x[1], batch_size=batch_size)
    x[2] = DataLoader(x[2], batch_size=batch_size)
    
    if type(x[0]) == tgd.batch.DataBatch:
        print('Graph drug data')
        x[0] = DataLoaderGeo(x[0], batch_size=batch_size)
    else:
        x[0] = DataLoader(x[0], batch_size=batch_size)
        
    return x


def create_smiles_hot(drugs, drug_to_econding_dict: dict):
    '''Create input data for drugs using smiles one-hot enconding
    
    '''

    x_drug_final = []
    for drug in drugs:
        x_drug_final.append(drug_to_econding_dict[drug])

    x_drug_final = np.dstack(x_drug_final)
    x_drug_final = np.rollaxis(x_drug_final, -1)

    return x_drug_final




def split(seed, _all_cls, _all_drugs, all_targets, train_size=0.8, 
          split_type='cblind'):
    '''Train test split for cancer or drug blind testing (no val set)
    
    Cancer blind testing means cell lines do not overlap between
    the train test and val sets. 
    Drug blind testing means drugs do not overlap between
    the train test and val sets. 
    '''
    #input cheks
    if type(_all_drugs) == type(_all_cls) != pd.Index:
        print('_all_drugs and _all_cls need to be PD indxes')

    #cancer blind splitting
    if split_type == 'cblind': 
        train_cls, test_cls = train_test_split(_all_cls, train_size=train_size, 
                                               random_state=seed)

        assert len(set(train_cls).intersection(test_cls)) == 0

        frac_train_cl = len(train_cls) / len(_all_cls)
        frac_test_cl = len(test_cls) / len(_all_cls)

        print('Fraction of cls in sets, relative to all cls'\
              'before mising values are removed')          
        print(f'train fraction {frac_train_cl}, test fraction {frac_test_cl}')
        print('------')


        #add in the drugs to each cell line. 
        def create_cl_drug_pair(cells):
            all_pairs = []
            for drug in _all_drugs:
                pairs = cells + '::' + drug
                #only keep pair if there is a truth value for it
                for pair in pairs:
                    if pair in all_targets:
                        all_pairs.append(pair)


            return np.array(all_pairs)

        train_pairs = create_cl_drug_pair(train_cls)
        test_pairs = create_cl_drug_pair(test_cls)
        
    #drug blind splitting    
    if split_type == 'dblind':
        train_ds, test_ds = train_test_split(_all_drugs, train_size=train_size, 
                                           random_state=seed)

        assert len(set(train_ds).intersection(test_ds)) == 0

        frac_train_ds = len(train_ds) / len(_all_drugs)
        frac_test_ds = len(test_ds) / len(_all_drugs)

        print('Fraction of drugs in sets, relative to all drugs'\
              'before mising values are removed')          
        print(f'train fraction {frac_train_ds}, test fraction {frac_test_ds}')
        print('------')

        #add in the drugs to each cell line. 
        def create_cl_drug_pair(drugs):
            all_pairs = []
            for cell in _all_cls:
                pairs = cell + '::' + drugs
                #only keep pair if there is a truth value for it
                for pair in pairs:
                    if pair in all_targets:
                        all_pairs.append(pair)


            return np.array(all_pairs)

        train_pairs = create_cl_drug_pair(train_ds)
        test_pairs = create_cl_drug_pair(test_ds)


    train_pairs = sklearn.utils.shuffle(train_pairs, random_state=seed)
    test_pairs = sklearn.utils.shuffle(test_pairs, random_state=seed)
          

    assert len(set(train_pairs).intersection(test_pairs)) == 0

    num_all_examples = len(_all_cls) * len(_all_drugs)
    frac_train_pairs = len(train_pairs) / num_all_examples
    frac_test_pairs = len(test_pairs) / num_all_examples
          
    print('Fraction of cls in sets, relative to all cl drug pairs, after '\
          'mising values are removed')
    print(f'train fraction {frac_train_pairs}, test fraction '\
          f'{frac_test_pairs}')

    #checking split works as expected.  
 
    if split_type == 'cblind':   
    #create mapping of cls to cl drug pairs for test train and val set. 
        def create_cl_to_pair_dict(cells):
            '''Maps a cell line to all cell line drug pairs with truth values

            '''
            dic = {}
            for cell in cells:
                dic[cell] = []
                for drug in _all_drugs:
                    pair = cell + '::' + drug
                    #filters out pairs without truth values
                    if pair in all_targets:
                        dic[cell].append(pair)
            return dic

        train_cl_to_pair = create_cl_to_pair_dict(train_cls)
        test_cl_to_pair = create_cl_to_pair_dict(test_cls)
        #check right number of cls 
        assert len(train_cl_to_pair) == len(train_cls)
        assert len(test_cl_to_pair) == len(test_cls)

    if split_type == 'dblind':
        def create_cl_to_pair_dict(drugs):
            '''Maps a drug to all cell line drug pairs with truth values

            '''
            dic = {}
            for drug in drugs:
                dic[drug] = []
                for cell in _all_cls:
                    pair = cell + '::' + drug
                    #filters out pairs without truth values
                    if pair in all_targets:
                        dic[drug].append(pair)
            return dic
        
        train_cl_to_pair = create_cl_to_pair_dict(train_ds)
        test_cl_to_pair = create_cl_to_pair_dict(test_ds)
        #check right number of drugs 
        assert len(train_cl_to_pair) == len(train_ds)
        assert len(test_cl_to_pair) == len(test_ds)
    
    # more checks 
    #unpack dict check no overlap and correct number 
    def flatten(l):
        return [item for sublist in l for item in sublist]

    train_flat = flatten(train_cl_to_pair.values())
    test_flat = flatten(test_cl_to_pair.values())

    assert len(train_flat) == len(train_pairs)
    assert len(test_flat) == len(test_pairs)
    assert len(set(train_flat).intersection(test_flat)) == 0
    
    return train_pairs, test_pairs



def plot_cv(train, val, epochs, err=2, skip_epochs=0,
            mm_loss = [], y_lab='Loss', 
            save_name=''):
    '''Func to plot the cross validation loss or metric
    
    Inputs
    ------
    train: list, of length number of cv folds, with each element of the list
    contaning a lists with the train set loss or metric. 
        
    val: same as train but with validation data  
    
    epochs: number of epochs model traiend for
    
    err: 0 1 or 2, defalt=2
    If 0, no error plotted
    If 1, error bars (s.d) plotted
    If 2, contionus error plotted
    
    skip_epochs: int, defalt=0
    number of epochs to skip at the start of plotting 
    
    y_lab: str, defalt=loss
    y label to plot
    
    save_name: str, defalt=''
    file_path\name to save fig. if defalt fig not saved
    '''
    x = range(1, epochs + 1 - skip_epochs) 
    val_mean = cv_metric(val, np.mean)
    train_mean = cv_metric(train, np.mean)
    val_sd = cv_metric(val, np.std)
    train_sd = cv_metric(train, np.std)
    if mm_loss:
        val_mm_mean = np.mean(mm_loss)
        val_mm_mean = np.array([val_mm_mean] * epochs)
        val_mm_sd = np.std(mm_loss)
        val_mm_sd = np.array([val_mm_sd] * epochs)
    
    if err == 1:
        plt.errorbar(
            x, train_mean[skip_epochs: ], yerr= train_sd[skip_epochs: ], 
            label='Train')
        plt.errorbar(
            x, val_mean[skip_epochs: ],  yerr = val_sd[skip_epochs: ], 
            label='Validation')
        plt.fill_between(
            x, train_mean[skip_epochs: ] - train_sd[skip_epochs: ], 
            yfit + train_sd[skip_epochs: ], color='gray', alpha=0.2)
        
        
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(y_lab)
        
    if err == 2:
        plt.plot(x, train_mean[skip_epochs: ], label='Train')
        plt.plot(x, val_mean[skip_epochs: ], label='Validation')
        plt.fill_between(x, train_mean[skip_epochs: ] - train_sd[skip_epochs: ], 
                         train_mean[skip_epochs: ] + train_sd[skip_epochs: ], 
                         color='gray', alpha=0.8)
        plt.fill_between(x, val_mean[skip_epochs: ] - val_sd[skip_epochs: ], 
                 val_mean[skip_epochs: ] + val_sd[skip_epochs: ], 
                 color='gray', alpha=0.8)
        if mm_loss:
            plt.plot(x, val_mm_mean[skip_epochs: ], label='Mean')
            plt.fill_between(x, val_mm_mean[skip_epochs: ] - val_mm_sd[skip_epochs: ],
                             val_mm_mean[skip_epochs: ] + val_mm_sd[skip_epochs: ],
                             alpha=0.6)

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(y_lab)        
        
    else: 
        plt.plot(x, train_mean[skip_epochs: ], label='Train')
        plt.plot(x, val_mean[skip_epochs: ], label='Validation')
        plt.plot(x, val_mm_mean[skip_epochs: ], label='Mean')
        plt.legend()
        
    if save_name:
        plt.savefig(save_name, dpi=1000)
    
    *_, best_epoch = best_metric(val)
    plt.axvline(1 + best_epoch - skip_epochs)
        
    print(best_metric(val))
        
    plt.show()