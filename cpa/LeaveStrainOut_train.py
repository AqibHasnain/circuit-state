import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from _model import AutoEncoder, PertEncoder, Discriminator
from _utils import MinMaxScale

# Model ID
if 'model_stats.csv' not in os.listdir('model_stats'): # no saved reports (not necessarily models)
    model_id = 'model_0000'
else: 
    saved_model_names = pd.read_csv('model_stats/model_stats.csv',index_col=0).index.tolist()
    model_nums = [int(x.split('_')[1]) for x in saved_model_names]
    model_id = 'model_' + '%04d' % (max(model_nums) + 1) 
print(model_id)

save_report = True
NUM_EPOCHS = 25

# randomly set hyperparameters before fixing random seeds
lam = np.random.uniform(1e-5,1e-2) # fraction of discriminator loss to use as reward for autoencoder reconstruction loss
time_ara_iptg_range = np.random.uniform(1,10)
print(f'Fraction of discriminator loss to use as reward of autoencoder recon loss, lam: {lam:.3f}')
print(f'Feature maximum for time, ara, iptg covariates: {time_ara_iptg_range}')

# set random seeds
np.random.seed(42)
torch.manual_seed(42)

# device to train model on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# load data
adata = pickle.load(open('odata_NANDiterate.pkl','rb'))

# get host gene data
X = torch.tensor(adata.X[:,0:-9]).type(torch.float) # host gene transcriptomics (already log1p'd)

# form perturbation data matrix
# continuous perturbations (circuit gene expression, time, IPTG_dose, Ara_dose)
pert = adata.X[:,-9:] # circuit genes (already log1p'd)
times = adata.obs.timepoint.astype(np.float32).values.reshape(-1,1)
times = MinMaxScale(times,feature_range=(0,time_ara_iptg_range))
adata.obs.timepoint = times
iptg = adata.obs.IPTG_concentration.astype(np.float32).values.reshape(-1,1) 
iptg = MinMaxScale(iptg,feature_range=(0,time_ara_iptg_range))
adata.obs.IPTG_concentration = iptg
ara = adata.obs.arabinose_concentration.astype(np.float32).values.reshape(-1,1) 
ara = MinMaxScale(ara,feature_range=(0,time_ara_iptg_range))
adata.obs.arabinose_concentration = ara
pert = np.concatenate((pert,times,iptg,ara),axis=1)
pert = torch.tensor(pert).type(torch.float)


# Network params 
input_dim = X.shape[1]
pert_dim = pert.shape[1]
hidden_dim = 128
basal_dim = 2


# in split loop to reset weights
autoenc = Autoencoder(input_dim, [hidden_dim, basal_dim], pert_dim)
autoenc.to(device)
discrim = Discriminator(basal_dim, pert_dim)
discrim.to(device)
pertdec = PertDecoder(pert_dim, basal_dim)
pertdec.to(device)

criterion = nn.MSELoss()

autoenc_optimizer = torch.optim.Adam(autoenc.parameters())
discrim_optimizer = torch.optim.Adam(discrim.parameters())
pertdec_optimizer = torch.optim.Adam(pertdec.parameters())

