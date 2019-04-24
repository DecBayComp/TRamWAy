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
Temporary file where stuff usefull for supervised classification is put,
to try and save datasets.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import tqdm
from torch import nn, optim
from torch.nn import functional as F

from .rw_features import *


def interpolate(X1, X2, alpha):
    return alpha * X1 + (1-alpha) * X2


def get_interpolation(t, x, tnew, old=False):
    N = len(t)
    Nnew = len(tnew)
    xnew = np.zeros((Nnew, x.shape[1]))
    dt = t[1] - t[0]
    dtnew = tnew[1] - tnew[0]
    for i in range(Nnew):
        if old:
            j = min(int(dtnew / dt * i), N-2)
            alpha = (t[j+1] - tnew[i]) / dt
            xnew[i] = interpolate(x[j], x[j+1], alpha)
        else:
            j = min(int(dtnew / dt * i), N-1)
            xnew[i] = x[j]
    return xnew


def normalize_length(RW, N):
    tnew = np.linspace(0, RW.t.max(), num=N, endpoint=True)
    Xnew = get_interpolation(RW.t.values, RW.loc[:, ['x', 'y']].values, tnew)
    data = np.stack((tnew,) + tuple(Xnew[:, i] for i in range(2))).T
    return pd.DataFrame(data=data, columns=['t'] + ['x', 'y'])


def process_rw(args):
    rw, id_, kwargs = args
    length_rw = len(rw)
    dt = rw.t.iloc[1] - rw.t.iloc[0]
    X = rw.loc[:, ['x', 'y']].values
    t = rw.t.values
    rw_obj = RandomWalk(rw)
    steps = np.diagonal(rw_obj.Dabs, offset=1)
    mean_step = np.mean(steps)
    sum_step = np.sum(steps)
    cum_dist = np.insert(
        np.cumsum(steps / sum_step), 0, 0)
    tau, msd = temporal_msd(
        rw_obj, n_samples=kwargs['wmax'], use_all=False)
    n_msd = len(msd)
    RW_ws = {}
    scale_prm = {}
    for k, w in enumerate(kwargs['windows']):
        RW_w = normalize_length(rw, w)
        RW_w.loc[:, ['x', 'y']] /= mean_step
        RW_w['x_perc'] = RW_w.loc[:, 'x'] / sum_step
        RW_w['y_perc'] = RW_w.loc[:, 'y'] / sum_step
        interp_f = scipy.interpolate.interp1d(
            t, cum_dist, fill_value=(min(cum_dist), max(cum_dist)))
        RW_w['cum_dist'] = interp_f(RW_w.t.values)
        interp_f = scipy.interpolate.interp1d(
            tau, msd, fill_value=(min(msd), max(msd)))
        RW_w['msd_log'] = interp_f(np.linspace(min(tau), max(tau), num=w))
        RW_ws[w] = RW_w.loc[:, [
            'x', 'y', 'x_perc', 'y_perc', 'cum_dist', 'msd_log']].values.T
        scale_prm[w] = {'dt': dt * length_rw / w}
    meta_prm = {'Tmax': rw.t.max(), 'mean_step': mean_step,
                'sum_step': sum_step}
    return RW_ws, scale_prm, meta_prm, id_


class get_label_from_func():
    def __init__(self, types):
        self.types_name = list(set([x.__name__ for x, _ in types]))
        nt = len(self.types_name)
        self.labels = []
        for i in range(len(self.types_name)):
            v = np.zeros(nt)
            v[i] = 1
            self.labels.append(v)
        self.dict_val_label = dict(list(zip(self.types_name, self.labels)))

    def __call__(self, prm):
        return self.dict_val_label[prm['func_name']]


class RWDataset(torch.utils.data.Dataset):
    """torch Dataset subclass.
    """

    def __init__(self, RWs, prms, windows=[100, 10], nb_process=4,
                 y_func='from_func', **kwargs):
        self.RWs = RWs
        self.RWsgroup = RWs.groupby('n')
        self.prms = prms
        self.windows = windows
        self.w_max = max(windows)
        self.trajs = RWs.n.unique()
        self.dict_index_traj = dict(
            list(zip(np.arange(len(self.trajs)), self.trajs)))
        self.RW_ws, self.scale_prm = dict(
            (w, {}) for w in windows), dict((w, {}) for w in windows)
        self.meta_prm = {}
        if y_func == 'from_func':
            self.get_y_func = get_label_from_func(kwargs['types'])

        rws = [(rw, id_, {'windows': self.windows, 'wmax': self.w_max})
               for id_, rw in self.RWs.groupby('n')]
        desc = 'distributed work'
        n = len(self.trajs)
        if nb_process is None:
            raw_data = list(map(process_rw,
                                tqdm.tqdm_notebook(rws, total=n, desc=desc)))
        else:
            with mp.Pool(nb_process) as p:
                raw_data = list(tqdm.tqdm_notebook(p.imap(process_rw, rws),
                                                   total=n, desc=desc))

        for i in tqdm.tqdm_notebook(range(len(raw_data)), desc='concat'):
            RW_ws_i, scale_prm_i, meta_prm_i, id_ = raw_data[i]
            self.meta_prm[id_] = meta_prm_i
            for k, w in enumerate(self.windows):
                self.RW_ws[w][id_] = RW_ws_i[w]
                self.scale_prm[w][id_] = scale_prm_i[w]

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.dict_index_traj[index]
        rws = tuple([torch.tensor(self.RW_ws[w][traj]).float()
                     for w in self.windows])
        scale_prms = tuple([torch.tensor(
            list(self.scale_prm[w][traj].values())).float()
            for w in self.windows])
        meta_prms = torch.tensor(list(self.meta_prm[traj].values())).float()
        y = self.get_y_func(self.prms[traj])
        return rws, scale_prms, meta_prms, torch.tensor(y).float()


class conv_layer(nn.Module):
    def __init__(self, w, scale_size=1, out_size=50, input_channel=1):
        super(conv_layer, self).__init__()
        self.w = w
        k_size = 7 if (self.w > 50) else (
            5 if (self.w < 50 and self.w > 15) else 3)
        wtemp = self.w
        temp_conv = []
        channels_in = input_channel
        channels_out = 16
        while wtemp > k_size:
            temp_conv.extend([nn.Conv1d(in_channels=channels_in,
                                        out_channels=channels_out,
                                        kernel_size=k_size),
                              nn.ReLU(), nn.MaxPool1d(2)])
            wtemp = (wtemp - 2 * k_size // 2) / 2
            channels_in = channels_out
            channels_out *= 2
        temp_conv.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*temp_conv)
        self.fc_scaler = nn.Linear(channels_in + scale_size, out_size)

    def forward(self, RW, scale_prm):
        out_rw = torch.squeeze(self.conv(RW), dim=2)
        out_rw.view(out_rw.size(0), -1)
        return F.relu(self.fc_scaler(torch.cat((out_rw, scale_prm), dim=1)))


class RWNet(nn.Module):
    def __init__(self, hidden_dims, supervised=True, windows=[100, 10],
                 scale_size=1, out_cnn_size=40, meta_size=2, input_channels=6,
                 output_size_softmax=1, output_size_linear=0):
        super(RWNet, self).__init__()
        self.hidden_dims = hidden_dims
        self.output_size_softmax = output_size_softmax
        self.output_size_linear = output_size_linear
        self.output_size = self.output_size_softmax + self.output_size_linear
        self.supervised = supervised
        self.windows = windows
        self.input_channels = input_channels

        self.conv_layers = {}
        for w in self.windows:
            for channel in range(self.input_channels):
                self.conv_layers[(w, channel)] = conv_layer(
                    w, scale_size=scale_size, out_size=out_cnn_size)

        enc_layers = [nn.Linear((out_cnn_size * len(self.windows) *
                                 self.input_channels) + meta_size,
                                self.hidden_dims[0]),
                      nn.ReLU()]
        lin_enc = [nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])
                   for i in range(len(self.hidden_dims)-1)]
        for i in range(len(lin_enc)):
            enc_layers += [lin_enc[i], nn.ReLU()]
        self.fc = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.output_size)
        if not supervised:
            self.fc_logvar = nn.Linear(self.hidden_dims[-1], self.output_size)

    def forward(self, rws, scale_prms, meta_prms):
        feat_in_fc = torch.cat((tuple(
            self.conv_layers[(w, channel)](rws[i][:, [channel], :],
                                           scale_prms[i])
            for i, w in enumerate(self.windows)
            for channel in range(self.input_channels)) +
            (meta_prms,)), dim=1)
        feat_out_fc = self.fc(feat_in_fc)
        if self.supervised:
            if self.output_size_linear == 0:
                return torch.squeeze(F.softmax(self.fc_mu(feat_out_fc), dim=1))
            else:
                raw_activ = self.fc_mu(feat_out_fc)
                return torch.cat((
                    F.softmax(raw_activ[:, :self.output_size_softmax]),
                    raw_activ[:, self.output_size_softmax:]), dim=1)

        else:
            return self.fc_mu(feats), self.fc_logvar(feats)
