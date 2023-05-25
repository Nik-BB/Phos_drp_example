import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
device = "cuda" if torch.cuda.is_available() else "cpu" 
def eval_torch(test_dls, model):
    model.eval()
    preds = []
    with torch.no_grad():
        for xo, xd, y in zip(test_dls[0], test_dls[1], test_dls[2]) :
            xo, xd = xo.to(device), xd.to(device)
            pred = model(xo, xd)
            pred = pred.cpu().detach().numpy()
            pred = pred.reshape(len(pred))
            preds.extend(pred)
    return np.array(preds)

def find_metrics(pred, true, m_list=['PCC', 'Rho', 'MSE', 'R2']):
    metric_dict = {}
    for metric in m_list:
        if metric == 'PCC':
            p = pearsonr(true, pred)[0]
            metric_dict['PCC'] = p
        if metric == 'MSE':
            mse = mean_squared_error(true, pred)
            metric_dict['MSE'] = mse
        if metric == 'R2':
            r2 = r2_score(true, pred)
            metric_dict['R2'] = r2
        if metric == 'Rho':
            rho = spearmanr(true, pred)[0]
            metric_dict['Rho'] = rho
            
    return metric_dict