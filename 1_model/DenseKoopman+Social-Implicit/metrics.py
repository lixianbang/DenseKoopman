import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx


def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0])**2 +
                                  (pred[i, t, 1] - target[i, t, 1])**2)
        sum_all += sum_ / (N * T)

    return sum_all / All


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0])**2 +
                                  (pred[i, t, 1] - target[i, t, 1])**2)
        sum_all += sum_ / (N)

    return sum_all / All


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]  #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :],
                                       axis=0) + init_node[ped, :]

    return nodes_.squeeze()


########################################################################################################################
def ade_kpspace(predAll, targetAll, count_):
    All = len(predAll)

    all_pred = np.zeros((20, 20, 2))
    all_target = np.zeros((20, 20, 2))

    n = np.swapaxes(predAll[0][:, :count_[0], :], 0, 1).shape[0]
    t = np.swapaxes(predAll[0][:, :count_[0], :], 0, 1).shape[1]

    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]

        for i in range(N):
            for t in range(T):
                all_pred[i, t, 0] += pred[i, t, 0]
                all_pred[i, t, 1] += pred[i, t, 1]

                all_target[i, t, 0] += target[i, t, 0]
                all_target[i, t, 1] += target[i, t, 1]
    sum_ade = 0
    for ix in range(n):
        for tx in range(t):

            sum_ade += math.sqrt((all_pred[ix, tx, 0] - all_target[ix, tx, 0])**2 +
                                 (all_pred[ix, tx, 1] - all_target[ix, tx, 1])**2)

    ade_output = sum_ade / (n * t)

    return ade_output


def fde_kpspace(predAll, targetAll, count_):
    All = len(predAll)

    all_pred = np.zeros((20, 20, 2))
    all_target = np.zeros((20, 20, 2))

    n = np.swapaxes(predAll[0][:, :count_[0], :], 0, 1).shape[0]
    t = np.swapaxes(predAll[0][:, :count_[0], :], 0, 1).shape[1]

    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]

        for i in range(N):
            for t in range(T):
                all_pred[i, t, 0] += pred[i, t, 0]
                all_pred[i, t, 1] += pred[i, t, 1]

                all_target[i, t, 0] += target[i, t, 0]
                all_target[i, t, 1] += target[i, t, 1]
    sum_fde = 0
    for ix in range(n):
        for tx in range(t-1, t):
            sum_fde += math.sqrt((all_pred[ix, tx, 0] - all_target[ix, tx, 0]) ** 2 +
                                 (all_pred[ix, tx, 1] - all_target[ix, tx, 1]) ** 2)

    fde_output = sum_fde / n

    return fde_output