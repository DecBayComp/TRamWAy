# -*- coding:utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#    Contributor: Maxime Duval

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""
Here are defined useful functions for training the unsupervised model.
"""

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import tqdm


class TabDataset(torch.utils.data.Dataset):
    """torch Dataset subclass. Performs 0 mean and 1 variance on features.
    """

    def __init__(self, df, c_drop={}):
        self.Features = list(set(df.columns).difference(c_drop))
        self.df = df.loc[:, self.Features].dropna().sort_index(axis=1)
        self.dict_iloc_to_traj = dict(
            list(zip(np.arange(len(self.df)), self.df.index)))
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(self.df)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.scaler.transform(
            (self.df.iloc[index].values)
            .reshape(1, -1))).float()


class VAE(nn.Module):
    """
    Neural network as seen in Kingma, D. P., & Welling, M. (2013).
    Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

    Parameters
    ----------
    input_dim : int, dimension of vectors we want to compress.
    hidden_dims : list of int, dimension(s) of the hidden layers of the neural
        network.
    latent_dim : int, dimension of the latent space on which we compress input
        vectors.
    ps : list of float between 0 and 1, dropout rates at each layer. Should
        have the same dimension as hidden_dims.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, ps):
        super(VAE, self).__init__()
        assert len(hidden_dims) == len(ps), 'dropouts and hidden dims must'
        'have same dimension'
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        range_h = range(len(self.hidden_dims)-1)
        self.latent_dim = latent_dim
        self.ps = ps

        self.fc_in = nn.Linear(self.input_dim, self.hidden_dims[0])
        lin_enc = [nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])
                   for i in range_h]
        enc_layers = []
        for i in range(len(lin_enc)):
            enc_layers += [lin_enc[i], nn.ReLU(), nn.Dropout(p=self.ps[i+1])]
        self.fc_enc = nn.Sequential(*enc_layers)

        self.fc_latin1 = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_latin2 = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_latout = nn.Linear(self.latent_dim, self.hidden_dims[-1])

        lin_dec = [nn.Linear(self.hidden_dims[i+1], self.hidden_dims[i])
                   for i in reversed(range_h)]
        dec_layers = []
        for i in range(len(lin_enc)):
            dec_layers += [lin_dec[i], nn.ReLU(), nn.Dropout(p=self.ps[-i-2])]
        self.fc_dec = nn.Sequential(*dec_layers)

        self.fc_out = nn.Linear(self.hidden_dims[0], self.input_dim)

    def encode(self, x):
        h1 = F.dropout(F.relu(self.fc_in(x)), p=self.ps[0])
        h1 = self.fc_enc(h1)
        return self.fc_latin1(h1), self.fc_latin2(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.dropout(F.relu(self.fc_latout(z)), p=self.ps[-1])
        h3 = self.fc_dec(h3)
        # return torch.sigmoid(self.fc_out(h3))
        return self.fc_out(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function_vae(beta=1):
    def tmp_func(recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = - beta * 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD
    return tmp_func


def train_vae(model, optimizer, loss_fct, data_loader, device,
              nb_epochs=3, disp_per_epoch=10):
    for epoch in tqdm.tqdm_notebook(range(nb_epochs)):
        model.train()
        train_loss = 0
        avg_loss = 0
        BCE_loss = 0
        KLD_loss = 0
        for batch_idx, data in tqdm.tqdm_notebook(enumerate(data_loader),
                                                  total=len(data_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, BCE, KLD = loss_fct(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            BCE_loss += BCE
            KLD_loss += KLD
            optimizer.step()
            if (batch_idx % int(len(data_loader) / disp_per_epoch) ==
                    int(len(data_loader) / disp_per_epoch) - 1):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader),
                    train_loss / int(len(data_loader) / disp_per_epoch)))
                print('Mean BCE : {0:.6f}, Mean KLD : {1:.6f}'.format(
                    (BCE_loss / int(len(data_loader) / disp_per_epoch)),
                    (KLD_loss / int(len(data_loader) / disp_per_epoch))))
                avg_loss += train_loss
                train_loss = 0
                BCE_loss = 0
                KLD_loss = 0
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch+1, avg_loss / len(data_loader)))


def get_mu_logvar(dl, model, device):
    mus = []
    logvars = []
    for batch_idx, data in tqdm.tqdm_notebook(enumerate(dl), total=len(dl)):
        data = data.to(device)
        mu, logvar = model.encode(data)
        mus.append(mu.cpu().detach().numpy()[:, 0, :])
        logvars.append(logvar.cpu().detach().numpy()[:, 0, :])
    mus = np.concatenate(mus)
    logvars = np.concatenate(logvars)
    return mus, logvars
