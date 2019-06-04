# -*- coding:utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#    Contributor: Maxime Duval

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d as mp3d

from .batch_generation import create_batch_rw


def visualize_random_walk(RW, color=True, colorbar=True):
    dim = len(set(RW.columns).intersection({'x', 'y', 'z'}))
    if dim == 1:
        fig, ax = plt.subplots()
        if color:
            points = np.array([RW.t, RW.x]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(RW.t.min(), RW.t.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(RW.t)
            line = ax.add_collection(lc)
            ax.axis('square')
            if colorbar:
                cbar = plt.colorbar(line, ax=ax)
                cbar.set_label('time')
        else:
            line = ax.plot(RW.x)
    elif dim == 2:
        fig, ax = plt.subplots()
        if color:
            points = np.array([RW.x, RW.y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(RW.t.min(), RW.t.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(RW.t)
            line = ax.add_collection(lc)
            ax.axis('square')
            if colorbar:
                cbar = plt.colorbar(line, ax=ax)
                cbar.set_label('time')
        else:
            line = ax.plot(RW.x, RW.y)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if color:
            points = np.array([RW.x, RW.y, RW.z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(RW.t.min(), RW.t.max())
            lc = mp3d.art3d.Line3DCollection(segments, cmap='viridis',
                                             norm=norm)
            lc.set_array(RW.t)
            line = ax.add_collection(lc)
            rmin, rmax = np.min(points), np.max(points)
            ax.set_xlim(rmin, rmax)
            ax.set_ylim(rmin, rmax)
            ax.set_zlim(rmin, rmax)
            if colorbar:
                cbar = plt.colorbar(line, ax=ax)
                cbar.set_label('time')
        else:
            ax.plot(RW.x, RW.y, RW.z)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')


def visualize_rw_hmm(RW):
    dim = len(set(RW.columns).intersection({'x', 'y', 'z'}))
    nstates = RW.state.max()
    dict_state_color = {}
    for k in range(int(nstates)+1):
        dict_state_color[k] = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][k]

    if dim == 1:
        fig, ax = plt.subplots()
        points = np.array([RW.t, RW.x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, nstates)
        lc = LineCollection(segments,
                            colors=[dict_state_color[x] for x in RW.state])
        line = ax.add_collection(lc)
        ax.axis('square')
        manual_legend = [Line2D([0], [0], marker='_', markersize=6, color=v,
                                label=f'diffusion type {k+1}')
                         for k, v in dict_state_color.items()]
        ax.legend(handles=manual_legend, loc=0)
    elif dim == 2:
        fig, ax = plt.subplots()
        points = np.array([RW.x, RW.y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, nstates)
        lc = LineCollection(segments,
                            colors=[dict_state_color[x] for x in RW.state])
        line = ax.add_collection(lc)
        ax.axis('square')
        manual_legend = [Line2D([0], [0], marker='_', markersize=6, color=v,
                                label=f'diffusion type {k+1}')
                         for k, v in dict_state_color.items()]
        ax.legend(handles=manual_legend, loc=0)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        points = np.array([RW.x, RW.y, RW.z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, nstates)
        lc = mp3d.art3d.Line3DCollection(segments,
                                         colors=[dict_state_color[x]
                                                 for x in RW.state])
        line = ax.add_collection(lc)
        ax.axis('square')
        manual_legend = [Line2D([0], [0], marker='_', markersize=6, color=v,
                                label=f'diffusion type {k+1}')
                         for k, v in dict_state_color.items()]
        ax.legend(handles=manual_legend, loc=0)
        rmin, rmax = np.min(points), np.max(points)
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(rmin, rmax)
        ax.set_zlim(rmin, rmax)


def plot_convex_hull(XY, hull, imax):
    """Plots a convex hull
    """
    plt.figure()
    plt.plot(XY[:, 0], XY[:, 1], linewidth=0.5)
    for simplex in hull.simplices[:-1]:
        plt.plot(XY[simplex, 0], XY[simplex, 1], 'r-')
    plt.plot(XY[hull.simplices[-1], 0],
             XY[hull.simplices[-1], 1], 'r-', label='Convex hull')
    plt.scatter(XY[hull.vertices, 0], XY[hull.vertices, 1], s=10,
                color='red', label='Nodes of the convex hull')
    # max_distance
    n = len(hull.vertices)
    p1, p2 = imax//n, imax % n
    plt.plot(XY[hull.vertices][[p1, p2], 0], XY[hull.vertices]
             [[p1, p2], 1], 'r--', label='max distance')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')


def plot_embedding_with_color(mu, logvar, color, plot_every=10, name=None,
                              square=False, s=0.1, alpha=1, figsize=(9, 6)):
    fig, axs = plt.subplots(figsize=figsize, ncols=2)
    axs[0].scatter(mu[::plot_every, 0], mu[::plot_every, 1],
                   c=color[::plot_every], s=s, alpha=alpha)
    if square:
        axs[0].axis('square')
    axs[0].set_title('mu')
    im = axs[1].scatter(logvar[::plot_every, 0], logvar[::plot_every, 1],
                        c=color[::plot_every], s=s, alpha=alpha)
    axs[1].set_title('logvar')
    if square:
        axs[1].axis('square')
    fig.suptitle(name)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)


def plot_embedding_classes(mu, logvar, dict_type_index, prms, dl,
                           plot_every=10, separate_ax=False, square=False,
                           s=0.1, alpha=1, dim=2, ax_logvar=0, figsize=(9, 6)):
    types = dict_type_index.keys()

    if dim == 2:
        if separate_ax:
            mu_min, mu_max = np.min(mu, axis=0), np.max(mu, axis=0)
            logvar_min = np.min(logvar, axis=0)
            logvar_max = np.max(logvar, axis=0)
            plt.figure(figsize=figsize)
            for i, type_ in enumerate(types):
                plt.subplot(2, len(types), i+1)
                plt.scatter(mu[dict_type_index[type_], 0][::plot_every],
                            mu[dict_type_index[type_], 1][::plot_every],
                            s=s, alpha=alpha)
                plt.axis((mu_min[0], mu_max[0], mu_min[1], mu_max[1]))
                plt.title(f'mu {type_}')
            for i, type_ in enumerate(types):
                plt.subplot(2, len(types), i+1+len(types))
                plt.scatter(logvar[dict_type_index[type_], 0][::plot_every],
                            logvar[dict_type_index[type_], 1][::plot_every],
                            s=s, alpha=alpha)
                plt.axis((logvar_min[0], logvar_max[0],
                          logvar_min[1], logvar_max[1]))
                plt.title(f'logvar {type_}')
            plt.tight_layout()
        else:
            plt.figure(figsize=figsize)
            plt.subplot(121)
            for type_ in types:
                plt.scatter(mu[dict_type_index[type_], 0][::plot_every],
                            mu[dict_type_index[type_], 1][::plot_every],
                            label=type_, s=s, alpha=alpha)
            plt.legend(markerscale=1/s)
            if square:
                plt.axis('square')
            plt.title('Repartition of different types (mu)')
            plt.subplot(122)
            for type_ in types:
                plt.scatter(logvar[dict_type_index[type_], 0][::plot_every],
                            logvar[dict_type_index[type_], 1][::plot_every],
                            label=type_, s=s, alpha=alpha)
            plt.legend(markerscale=1/s)
            plt.title('Repartition of different types (logvars)')
            if square:
                plt.axis('square')
            plt.tight_layout()

    if dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        for type_ in types:
            ax.scatter(mu[dict_type_index[type_], 0][::plot_every],
                       mu[dict_type_index[type_], 1][::plot_every],
                       logvar[dict_type_index[type_],
                              ax_logvar][::plot_every],
                       label=type_, s=s, alpha=alpha)
        ax.set_xlabel('mu_x')
        ax.set_ylabel('mu_y')
        if ax_logvar == 0:
            ax.set_zlabel('logvar_x')
        else:
            ax.set_zlabel('logvar_y')
        ax.legend(markerscale=1/s)
        ax.set_title('Repartition of different types')


def visualize_types(rw_types, names, nb_pr_row=2, nb_row=2, scale=2,
                    figsize=(16, 7)):
    N = nb_pr_row * nb_row
    ncols = 2
    nrows = len(rw_types) // 2 + len(rw_types) % 2
    print('Generating random walks...')
    dict_type_rws = dict(zip(names,
                             [create_batch_rw(n=N, ps=None, nb_process=None,
                                              types=[type_], pbar=False)[0]
                              for type_ in rw_types]))
    fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    axs = np.atleast_2d(axs)
    for i_type, type_name in zip(list(range(len(rw_types))), names):
        rws = dict_type_rws[type_name]
        for i in range(N):
            rw = rws.loc[rws.n == i, ['x', 'y']].values
            pos = np.array([i//(nb_row)*scale, i % (nb_row)*scale])
            rw += pos
            axs[i_type//2, i_type % 2].plot(rw[:, 0], rw[:, 1])
        for i in range(nb_row+1):
            s2 = scale / 2
            axs[i_type//2, i_type % 2].plot([-s2, nb_pr_row*scale-s2],
                                            [(i-0.5)*scale, (i-0.5)*scale],
                                            'k--', alpha=0.5, linewidth=1)
        for i in range(nb_pr_row+1):
            axs[i_type//2, i_type % 2].plot([(i-0.5)*scale, (i-0.5)*scale],
                                            [-scale/2, nb_row*scale-scale/2],
                                            'k--', alpha=0.5, linewidth=1)
        axs[i_type//2, i_type % 2].set_title(type_name)
        axs[i_type//2, i_type % 2].axis('scaled')
    if len(rw_types) % 2 == 1:
        fig.delaxes(axs[(len(rw_types)-1)//2, 1])
    plt.tight_layout()


def plot_rw(ax, RW):
    points = np.array([RW.x, RW.y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(RW.t.min(), RW.t.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(RW.t)
    line = ax.add_collection(lc)


def plot_sample_RWs(RWs, ncols=4, nrows=4, figsize=(16, 4), scale=2,
                    scale_to_box=False):
    fig, ax = plt.subplots(figsize=figsize)
    ns = RWs.n.unique()
    permut = np.random.permutation(len(ns))
    N = ncols * nrows
    ns_chosen = ns[permut[:N]]
    for i, n_i in enumerate(ns_chosen):
        rw = RWs.loc[RWs.n == n_i].copy()
        if scale_to_box:
            div = np.max(np.array([-rw.x.min(), rw.x.max(),
                                   rw.y.max(), -rw.y.min()]))
            rw.x /= div
            rw.y /= div
        rw.x += (i % ncols) * scale
        rw.y += (i // ncols) * scale
        plot_rw(ax, rw)
    for i in range(ncols):
        plt.axvline((i + 0.5) * scale, linestyle='--')
    for i in range(nrows):
        plt.axhline((i + 0.5) * scale, linestyle='--')
    plt.axis('scaled')
