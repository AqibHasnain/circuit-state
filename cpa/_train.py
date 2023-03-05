import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from _utils import gen_modelID, calculate_qc_metrics, get_perts, get_covs, train_one_epoch, compute_metrics
from _model import Autoencoder, Discriminator, PertDecoder, CovDecoder

save_model = False

# HYPERPARAMETERS
HIDDEN_DIM = 256     # number of nodes in hidden layers 
BASAL_DIM = 5        # latent space dimension
WD_ae = 0.0          # weight decay for gene encoder, perturbation encoder, and gene decoder regularization
WD_p = 0.0          # weight decay for perturbation decoder regularization
WD_c = 0.0           # weight decay for discrete covariate decoder regularization
LR_ae = 2            # learning rate for updating gene encoder, perturbation encoder, and gene decoder parameters
LR_p = 1.2          # learning rate for updating perturbation decoder parameters
LR_c = 1.0           # learning rate for updating discrete covariate decoder parameters
LAM = 0.001          # discrimination loss maximization during autoencoder updates
BATCH_SIZE = 128     # amt of data pts seen by network per parameter update
GENE_PER = 0.95      # percentile of number of zero expression genes per sample to use as threshold for sample filtering
SAMPLE_PER = 0.05    # percentile of number of samples with zero expression (per gene) to use as threshold for gene filtering

NUM_EPOCHS = 15000 # number of epochs to train for

# generate unique model ID
model_id = gen_modelID(path='saved_models')
print(model_id)

# device to train model on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Load data
adata = pickle.load(open('/home/deepuser/Desktop/aqib/circuit-state/NAND_iterate/odata_NANDiterate.pkl','rb')) # log1p'd

# Preprocess data
samples2keep, genes2keep = calculate_qc_metrics(adata.X[:,:-9], within_gene_per=GENE_PER, within_sample_per=SAMPLE_PER)
adata = adata[samples2keep, genes2keep + [True]*9].copy() # [True]*9 is there to keep the synthetic genes

# Convert host gene expression to torch tensor
X = torch.tensor(adata.X[:,0:-9]).type(torch.float) # host gene transcriptomics

# Get pertubation matrix
adata, P = get_perts(adata, time_scale=(0,5), iptg_scale=(0,15), ara_scale=(0,15))
P = torch.tensor(P).type(torch.float)

# Get covariate matrix 
C = get_covs(adata)
C = torch.tensor(C).type(torch.float)

pert_names = ['Sensor_TetR',
              'Sensor_LacI',
              'Sensor_AraC',
              'Sensor_LuxR',
              'Circuit_PhlF',
              'KanR',
              'CmR',
              'YFP',
              'Circuit_IcaR',
              'time',
              'IPTG',
              'Ara']

torch.manual_seed(1)

# encoder input dimensions
input_dim = X.shape[1]
pert_dim = P.shape[1]
cov_dim = C.shape[1]

# model instantiation
autoenc = Autoencoder(input_dim, [HIDDEN_DIM, BASAL_DIM], pert_dim, cov_dim)
discrim = Discriminator(BASAL_DIM, pert_dim)
discrim_cov = Discriminator(BASAL_DIM, cov_dim)
pertdec = PertDecoder(pert_dim, BASAL_DIM)
covdec = CovDecoder(cov_dim, BASAL_DIM)
autoenc.to(device)
discrim.to(device)
discrim_cov.to(device)
pertdec.to(device)
covdec.to(device)

mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

autoenc_optimizer = torch.optim.Adadelta(autoenc.parameters(), lr=LR_ae, weight_decay=WD_ae)
discrim_optimizer = torch.optim.Adam(discrim.parameters())
discrim_cov_optimizer = torch.optim.Adam(discrim_cov.parameters())
pertdec_optimizer = torch.optim.Adadelta(pertdec.parameters(), lr=LR_p, weight_decay=WD_p)
covdec_optimizer = torch.optim.Adadelta(covdec.parameters(), lr=LR_c, weight_decay=WD_c)

### Start training ### 
for epoch in range(NUM_EPOCHS):

    loss_x, loss_dp, loss_dc, loss_p, loss_c = train_one_epoch(
        autoenc, discrim, discrim_cov, pertdec, covdec,
        autoenc_optimizer, discrim_optimizer, discrim_cov_optimizer, pertdec_optimizer, covdec_optimizer, 
        mse, ce, X, P, C, BATCH_SIZE, LAM, device)

    if epoch % (NUM_EPOCHS*0.05) == 0: 
        print( f'EPOCH: {epoch}, X_LOSS: {loss_x:.6f}, P_LOSS: {loss_p:.6f}, C_LOSS: {loss_c:.6f}, DP_LOSS: {loss_dp:.6f}, DC_LOSS: {loss_dc:.6f}')


autoenc.eval()
discrim.eval()
discrim_cov.eval()
pertdec.eval()
covdec.eval()
with torch.no_grad():
    r2_x, r2_p, acc_c, r2_dp, acc_dc = compute_metrics(autoenc, discrim, discrim_cov, pertdec, covdec, X, P, C, device)

# if save_model: 
#     torch.save(autoenc.state_dict(), 'saved_models/'+'autoenc_'+model_id+'.pt')
#     torch.save(pertdec.state_dict(), 'saved_models/'+'pertdec_'+model_id+'.pt')
#     torch.save(discrim.state_dict(), 'saved_models/'+'discrim_'+model_id+'.pt')

#     # collect hyperparams and metrics and write to csv
    #   collection = {    'hidden_dim':HIDDEN_DIM,
    #                     'basal_dim':BASAL_DIM,
    #                     'weight_decay_ae':WD_ae,
    #                     'weight_decay_p':WD_p,
    #                     'weight_decay_c':WD_c,                                            
    #                     'lr_ae':LR_ae,
    #                     'lr_p':LR_p,
    #                     'lr_c':LR_c,
    #                     'lam':LAM,
    #                     'batch_size':BATCH_SIZE,
    #                     'num_epochs':NUM_EPOCHS,
    #                     'r2_x': r2_x_test,
    #                     'r2_d': r2_d_test,
    #                     'r2_p': r2_p_test}

#     report = pd.DataFrame(collection, index=[model_id])

#     stats_dir = os.listdir('model_stats')
#     if 'model_report.csv' in stats_dir:
#         report.to_csv('model_stats/model_report.csv', mode='a', header=False)
#     else: 
#         report.to_csv('model_stats/model_report.csv')