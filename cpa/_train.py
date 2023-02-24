import numpy as np
import pandas as pd
import pickle
import os
from copy import deepcopy
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from _utils import MinMaxScale
from _model import Autoencoder, Discriminator, PertDecoder

save_model = True

# Model ID
if os.listdir('saved_models') == []: # no saved models exist
    model_id = 'model_0001'
else: 
    saved_model_names = os.listdir('saved_models')
    model_nums = [int((x.split('_')[1]).split('.')[0]) for x in saved_model_names]
    model_id = 'model_' + '%04d' % (max(model_nums) + 1) 
print(model_id)

# device to train model on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Load data
adata = pickle.load(open('/home/deepuser/Desktop/aqib/circuit-state/NAND_iterate/odata_NANDiterate.pkl','rb'))
X = torch.tensor(adata.X[:,0:-9]).type(torch.float) # host gene transcriptomics (already log1p'd)

# get pertubation matrix
# continuous perturbations (circuit gene expression, time, IPTG_dose, Ara_dose)
pert = adata.X[:,-9:] # circuit genes (already log1p'd)
times = adata.obs.timepoint.astype(np.float32).values.reshape(-1,1)
times = MinMaxScale(times,feature_range=(0,2))
adata.obs.timepoint = times
iptg = adata.obs.IPTG_concentration.astype(np.float32).values.reshape(-1,1) 
iptg = MinMaxScale(iptg,feature_range=(0,2))
adata.obs.IPTG_concentration = iptg
ara = adata.obs.arabinose_concentration.astype(np.float32).values.reshape(-1,1) 
ara = MinMaxScale(ara,feature_range=(0,2))
adata.obs.arabinose_concentration = ara
pert = np.concatenate((pert,times,iptg,ara),axis=1)
pert = torch.tensor(pert).type(torch.float)

torch.manual_seed(1)

input_dim = X.shape[1]
pert_dim = pert.shape[1]
hidden_dim = 128
basal_dim = 2

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

### Start training ### 
NUM_EPOCHS = 100000
lam = 0.001
BATCH_SIZE = 128
loss = {'recon':[],'discrim':[],'pert_recon':[]}
for epoch in range(NUM_EPOCHS):
    
    permutation = torch.randperm(X.shape[0])
    running_loss_x = 0.0
    running_loss_d = 0.0
    running_loss_p = 0.0
    
    for cnt, jj in enumerate(range(0, X.shape[0], BATCH_SIZE)):

        batch_inds = permutation[jj:jj+BATCH_SIZE] # grab batch 
        batch_x, batch_p = X[batch_inds].to(device), pert[batch_inds].to(device) 

        # train the discriminator
        discrim_optimizer.zero_grad()
        
        batch_z = autoenc.encoder(batch_x) # latent x
        p_hat = discrim(batch_z)
        discrim_loss = criterion(p_hat, batch_p)  
        discrim_loss.backward(retain_graph=True)                 

        # train the autoencoder and pertencoder
        autoenc_optimizer.zero_grad()

        X_hat = autoenc(batch_x,batch_p)
        autoenc_loss = criterion(X_hat, batch_x)  - lam * discrim_loss
        autoenc_loss.backward()

        discrim_optimizer.step()
        autoenc_optimizer.step()
        
        # train the pertdecoder
        pertdec_optimizer.zero_grad()
        
        p_latent = autoenc.pertencoder(batch_p)
        p_hat = pertdec(p_latent)
        pertdec_loss = criterion(p_hat, batch_p)
        pertdec_loss.backward()
        
        pertdec_optimizer.step()

        with torch.no_grad():
            running_loss_x += autoenc_loss.item()
            running_loss_d += discrim_loss.item()
            running_loss_p += pertdec_loss.item()
            
    if epoch % (NUM_EPOCHS*0.05) == 0: 
        loss['recon'].append( running_loss_x/(cnt+1) )
        loss['discrim'].append( running_loss_d/(cnt+1) )
        loss['pert_recon'].append( running_loss_p/(cnt+1) )
        print( f'EPOCH: {epoch}, D_LOSS: {running_loss_d/(cnt+1)}, X_LOSS: {running_loss_x/(cnt+1)}, '
                  f'P_LOSS: {running_loss_p/(cnt+1)}'  )


autoenc.eval()
discrim.eval()
with torch.no_grad():
    
    X_hat = autoenc(X.to(device), pert.to(device)).cpu().numpy()
    print(r2_score(X.numpy(),X_hat))

    p_hat = discrim(autoenc.encoder(X.to(device))).cpu().numpy()
    print(r2_score(pert.numpy(),p_hat))
    
    p_hat = pertdec(autoenc.pertencoder(pert.to(device))).cpu().numpy()
    print(r2_score(pert.numpy(),p_hat))

if save_model: 
    torch.save(autoenc.state_dict(), 'saved_models/'+'autoenc_'+model_id+'.pt')
    torch.save(pertdec.state_dict(), 'saved_models/'+'pertdec_'+model_id+'.pt')
    torch.save(discrim.state_dict(), 'saved_models/'+'discrim_'+model_id+'.pt')