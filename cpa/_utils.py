import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score
import torch

def gen_modelID(path='saved_models'):
    # Model ID
    if os.listdir(path) == []: # no saved models exist
        model_id = 'model_0001'
    else: 
        saved_model_names = os.listdir(path)
        model_nums = [int((x.split('_')[-1]).split('.')[0]) for x in saved_model_names]
        model_id = 'model_' + '%04d' % (max(model_nums) + 1) 
    print(model_id)
    return model_id
    
def gen_modelID_lso(path='model_stats'):
    stats_dir = os.listdir(path)
    if 'unseen_strain_report.csv' in stats_dir:
        model_nums = pd.read_csv(path+'/unseen_strain_report.csv',index_col=0).index.tolist()
        model_nums = [int((x.split('_')[-1]).split('.')[0]) for x in model_nums]
        model_id = 'lso_' + '%04d' % (max(model_nums) + 1)
    else: 
        model_id = 'lso_0001'
    return model_id


def MinMaxScale(arr,feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(arr)
    return scaler.transform(arr)


def calculate_qc_metrics(x, within_gene_per=0.95, within_sample_per=0.05):
    nZeroGenes = (~(x > 0.0)).sum(axis=1) # number of genes with zero expression in each sample
    nZeroGenes_cutoff = np.percentile(nZeroGenes, 100*within_gene_per)
    print('within sample cutoff', nZeroGenes_cutoff)
    print(f'Number of samples with high amount of zero genes: {sum(nZeroGenes > nZeroGenes_cutoff)}')

    nZeroSamples = (~(x > 0.0)).sum(axis=0) # number of samples in which each gene has zero expression
    nZeroSamples_cutoff = np.percentile(nZeroSamples, 100*within_gene_per)
    print('within gene cutoff', nZeroSamples_cutoff)
    print(f'Number of genes with high amount of zeros: {sum(nZeroSamples > nZeroSamples_cutoff)}')

    return list(nZeroGenes <= nZeroGenes_cutoff), list(nZeroSamples <= nZeroSamples_cutoff)


def get_perts(adata,scale=True,time_scale=(0,5),iptg_scale=(0,15),ara_scale=(0,15)):
    pert = adata.X[:,-9:] # circuit genes 
    times = adata.obs.timepoint.astype(np.float32).values.reshape(-1,1)
    iptg = adata.obs.IPTG_concentration.astype(np.float32).values.reshape(-1,1) 
    ara = adata.obs.arabinose_concentration.astype(np.float32).values.reshape(-1,1) 
    if scale:
        times = MinMaxScale(times,feature_range=time_scale)
        iptg = MinMaxScale(iptg,feature_range=iptg_scale)
        ara = MinMaxScale(ara,feature_range=ara_scale)
    pert = np.concatenate((pert,times,iptg,ara),axis=1)

    # Update adata with scaled covariates
    adata.obs.timepoint = times
    adata.obs.IPTG_concentration = iptg
    adata.obs.arabinose_concentration = ara

    return adata, pert


def get_covs(adata):
    C = np.zeros((adata.n_obs, 3)) # one-hots of dim 3 (wild-type, genome, plasmid)
    for ii in range(adata.n_obs):
        if adata.obs.strain[ii] == 'wild-type':
            C[ii] = np.array([1,0,0]).T
        elif (adata.obs.strain[ii] == 'pJ2007_IcaR') or (adata.obs.strain[ii] == 'pJ2007_PhlF'):
            C[ii] = np.array([0,0,1]).T
        else: 
            C[ii] = np.array([0,1,0]).T
    return C


def train_one_epoch(autoenc, discrim, discrim_cov, pertdec, covdec,
                    autoenc_optimizer, discrim_optimizer, discrim_cov_optimizer, pertdec_optimizer, covdec_optimizer,
                    criterion, criterion_cov, X, P, C, batch_size, lam, device):

    permutation = torch.randperm(X.shape[0])
    loss_x = 0.0
    loss_dp = 0.0
    loss_dc = 0.0
    loss_p = 0.0
    loss_c = 0.0

    for cnt, jj in enumerate(range(0, X.shape[0], batch_size)):

        batch_inds = permutation[jj:jj+batch_size] # grab batch 
        batch_x = X[batch_inds].to(device)
        batch_p = P[batch_inds].to(device) 
        batch_c = C[batch_inds].to(device)
        
        # train the discriminators
        discrim_optimizer.zero_grad()
        discrim_cov_optimizer.zero_grad()
        
        batch_z = autoenc.encoder(batch_x) # latent x

        p_hat = discrim(batch_z) # prediction of all perts by discriminator
        discrim_loss = criterion(p_hat, batch_p)  
        discrim_loss.backward(retain_graph=True) # retain graph to use the loss as a reward for autoencoder
        
        c_hat = discrim_cov(batch_z) # prediction of covariates by discriminator
        discrim_cov_loss = criterion_cov(c_hat, batch_c)
        discrim_cov_loss.backward(retain_graph=True)
        
        # train the autoencoder, pertencoder, covencoder
        autoenc_optimizer.zero_grad()

        X_hat = autoenc(batch_x, batch_p, batch_c)
        autoenc_loss = criterion(X_hat, batch_x)  - lam * discrim_loss - lam * discrim_cov_loss
        autoenc_loss.backward()

        discrim_optimizer.step()
        discrim_cov_optimizer.step()
        autoenc_optimizer.step()
        
        # train the pertdecoder and covdecoder
        pertdec_optimizer.zero_grad()
        covdec_optimizer.zero_grad()
        
        p_latent = autoenc.pertencoder(batch_p)
        p_hat = pertdec(p_latent)
        pertdec_loss = criterion(p_hat, batch_p)
        pertdec_loss.backward()
        pertdec_optimizer.step()

        c_latent = autoenc.covencoder(batch_c)
        c_hat = covdec(c_latent)
        covdec_loss = criterion_cov(c_hat, batch_c)
        covdec_loss.backward()
        covdec_optimizer.step()

        with torch.no_grad():
            loss_x += autoenc_loss.item()/(cnt+1)
            loss_dp += discrim_loss.item()/(cnt+1)
            loss_dc += discrim_cov_loss.item()/(cnt+1)
            loss_p += pertdec_loss.item()/(cnt+1)
            loss_c += covdec_loss.item()/(cnt+1)

    return loss_x, loss_dp, loss_dc, loss_p, loss_c


def compute_metrics(autoenc, discrim, discrim_cov, pertdec, covdec, X, P, C, device):
    X_hat = autoenc(X.to(device), P.to(device), C.to(device)).cpu().numpy()
    r2_x = r2_score(X.numpy(),X_hat)
    print(f'x recon r2: {r2_x:.6f}')

    p_hat = pertdec(autoenc.pertencoder(P.to(device))).cpu().numpy()
    r2_p = r2_score(P.numpy(),p_hat)
    print(f'p recon r2: {r2_p:.6f}')

    c_hat = covdec(autoenc.covencoder(C.to(device))).cpu().numpy()
    acc_c = balanced_accuracy_score(C.numpy().argmax(axis=1),c_hat.argmax(axis=1))
    print(f'c recon acc: {acc_c:.6f}')

    p_hat = discrim(autoenc.encoder(X.to(device))).cpu().numpy()
    r2_dp = r2_score(P.numpy(),p_hat)
    print(f'p discrim r2: {r2_dp:.6f}')

    c_hat = discrim_cov(autoenc.encoder(X.to(device))).cpu().numpy()
    acc_dc = balanced_accuracy_score(C.numpy().argmax(axis=1),c_hat.argmax(axis=1))
    print(f'c discrim acc: {acc_dc:.6f}')

    return r2_x, r2_p, acc_c, r2_dp, acc_dc