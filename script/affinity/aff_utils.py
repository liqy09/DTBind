import os
import pickle
import numpy as np
import torch
import math
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from torch.nn.utils import weight_norm
# from torch.nn.utils.parametrizations import weight_norm
import random
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# embedding selection function
def add_index(input_array, ebd_size):
    batch_size, n_vertex, n_nbs = np.shape(input_array)
    add_idx = np.array(list(range(0, (ebd_size) * batch_size, ebd_size)) * (n_nbs * n_vertex))
    add_idx = np.transpose(add_idx.reshape(-1, batch_size))
    add_idx = add_idx.reshape(-1)
    new_array = input_array.reshape(-1) + add_idx
    return new_array


# padding functions
def pad_label(arr_list, ref_list):
	N = ref_list.shape[1]
	a = np.zeros((len(arr_list), N))
	for i, arr in enumerate(arr_list):
		n = len(arr)
		a[i, 0:n] = arr
	return a


def pack1D(arr_list):
    arr_list = [np.array(x) if isinstance(x, list) else x for x in arr_list]
    max_len = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), max_len))
    for i, arr in enumerate(arr_list):
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        n = arr.shape[0]
        a[i, 0:n] = arr
    return a



def pack2D_2(arr_list):
    arr_list = [np.array(x) if isinstance(x, list) else x for x in arr_list]

    N = max(x.shape[0] for x in arr_list if x.ndim > 0)
    M = max(x.shape[1] for x in arr_list if x.ndim > 0)  # 保持最大邻居数量
    a = np.zeros((len(arr_list), N, M))

    for i, arr in enumerate(arr_list):
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        n = arr.shape[0]
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        m = arr.shape[1] if arr.ndim > 1 else 1
        a[i, 0:n, 0:m] = arr
    return a

def pack2D_3(arr_list):

    arr_list = [np.array(x) if isinstance(x, list) else x for x in arr_list]

    M = max(arr.max() for arr in arr_list if arr.ndim > 0)
    N = max(x.shape[0] for x in arr_list if x.ndim > 0)

    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        n = arr.shape[0]

        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=-1)
        a[i, 0:n, 0:M] = arr
    return a


def pack2D(arr_list, max_nb=6):
    arr_list = [np.array(x) if isinstance(x, list) else x for x in arr_list]

    N = max(x.shape[0] for x in arr_list if x.ndim > 0)
    M = max_nb
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):

        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        n = arr.shape[0]

        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        m = arr.shape[1] if arr.ndim > 1 else 1
        a[i, 0:n, 0:m] = arr
    return a


def get_mask(arr_list):
    arr_list = [np.array(x) if isinstance(x, list) else x for x in arr_list]
    # Proceed with finding the maximum shape[0] value
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        a[i, :arr.shape[0]] = 1
    return a


def batch_data_process(data):

    (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs, p_num_nbs_mar, a_fea, b_fea, d_anb, d_bnb, d_nbs, num_nbs_mar,
     affinity_labels, edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, d_bond_nei, p_nei_nbs, d_nei_nbs, p_nbs_mar, d_nbs_mar) = data


    pretrained_fea = pack2D_2(pretrained_fea)
    surface_fea = pack2D_2(surface_fea)
    p_edge_fea = pack2D_2(p_edge_fea)
    # p_adj = pack2D_2(p_adj)
    p_anb = pack2D_2(p_anb)
    p_bnb = pack2D_2(p_bnb)
    # p_nbs = pack1D(p_nbs)
    p_nbs_mask = pack2D_2(p_num_nbs_mar)
    # print(f"p_nbs_mask:{p_nbs_mask}")
    p_mask = get_mask(p_nbs)  # (len(arr_list), N)
    # print(f"p_mask:{p_mask}")

    a_fea = pack2D_2(a_fea)
    b_fea = pack2D(b_fea)
    d_anb = pack2D(d_anb)
    d_bnb = pack2D(d_bnb)
    # d_nbs = pack1D(d_nbs)
    # d_adj = pack2D(d_adj)
    d_nbs_mask = pack2D(num_nbs_mar)
    d_mask = get_mask(d_nbs)

    edge_residue_to_atom = pack2D_2(edge_residue_to_atom)
    edge_atom_to_residue = pack2D_2(edge_atom_to_residue)
    p_nei = pack2D(p_nei)
    d_nei = pack2D(d_nei)
    p_bond_nei = pack2D(p_bond_nei)
    d_bond_nei = pack2D(d_bond_nei)

    p_hetero_mask = pack2D(p_nbs_mar)
    # print(f"p_hetero_mask:{p_hetero_mask}")
    d_hetero_mask = pack2D(d_nbs_mar)

    # p_mask = Variable(torch.FloatTensor(p_mask)).cuda()
    pretrained_fea = Variable(torch.FloatTensor(pretrained_fea)).cuda()
    surface_fea = Variable(torch.FloatTensor(surface_fea)).cuda()
    p_edge_fea = Variable(torch.FloatTensor(p_edge_fea)).cuda()
    # p_adj = Variable(torch.LongTensor(p_adj)).cuda()
    p_anb = Variable(torch.LongTensor(p_anb)).cuda()
    p_bnb = Variable(torch.LongTensor(p_bnb)).cuda()
    p_nbs_mask = Variable(torch.FloatTensor(p_nbs_mask)).cuda()
    p_mask = Variable(torch.FloatTensor(p_mask)).cuda()
    # p_nbs = Variable(torch.LongTensor(p_nbs)).cuda

    a_fea = Variable(torch.FloatTensor(a_fea)).cuda()
    b_fea = Variable(torch.FloatTensor(b_fea)).cuda()
    d_anb = Variable(torch.LongTensor(d_anb)).cuda()
    d_bnb = Variable(torch.LongTensor(d_bnb)).cuda()
    # d_nbs = Variable(torch.LongTensor(d_nbs)).cuda
    d_nbs_mask = Variable(torch.FloatTensor(d_nbs_mask)).cuda()
    # d_adj = Variable(torch.FloatTensor(d_adj)).cuda
    d_mask = Variable(torch.FloatTensor(d_mask)).cuda()

    edge_residue_to_atom = Variable(torch.FloatTensor(edge_residue_to_atom)).cuda()
    edge_atom_to_residue = Variable(torch.FloatTensor(edge_atom_to_residue)).cuda()
    p_nei = Variable(torch.LongTensor(p_nei)).cuda()
    d_nei = Variable(torch.LongTensor(d_nei)).cuda()
    p_bond_nei = Variable(torch.LongTensor(p_bond_nei)).cuda()
    d_bond_nei = Variable(torch.LongTensor(d_bond_nei)).cuda()

    p_hetero_mask = Variable(torch.FloatTensor(p_hetero_mask)).cuda()
    d_hetero_mask = Variable(torch.FloatTensor(d_hetero_mask)).cuda()


    return (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs_mask, p_mask, a_fea, b_fea, d_anb, d_bnb, d_nbs_mask, d_mask, affinity_labels,
            edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, d_bond_nei, p_hetero_mask, d_hetero_mask)


def data_from_index(data_pack, idx_list):

    pretrained_fea = data_pack[0][idx_list]
    surface_fea = data_pack[1][idx_list]
    p_edge_fea = data_pack[2][idx_list]
    # p_adj = data_pack[3][idx_list]
    p_anb = data_pack[3][idx_list]
    p_bnb = data_pack[4][idx_list]
    p_nbs = data_pack[5][idx_list]
    p_num_nbs_mar = data_pack[6][idx_list]
    a_fea = data_pack[7][idx_list]
    b_fea = data_pack[8][idx_list]
    d_anb = data_pack[9][idx_list]
    d_bnb = data_pack[10][idx_list]
    d_nbs = data_pack[11][idx_list]
    num_nbs_mar = data_pack[12][idx_list]
    # (surface_fea, p_edge_fea, p_adj, p_anb, p_bnb, p_nbs, a_fea, b_fea, d_anb, d_bnb, d_nbs) = [data_pack[i][idx_list] for i in range(12)]
    affinity_labels = data_pack[13][idx_list]
    edge_residue_to_atom = data_pack[14][idx_list]
    edge_atom_to_residue = data_pack[15][idx_list]
    p_nei = data_pack[16][idx_list]
    d_nei = data_pack[17][idx_list]
    p_bond_nei = data_pack[18][idx_list]
    d_bond_nei = data_pack[19][idx_list]
    p_nei_nbs = data_pack[20][idx_list]
    d_nei_nbs = data_pack[21][idx_list]
    p_nbs_mar = data_pack[22][idx_list]
    d_nbs_mar = data_pack[23][idx_list]


    return (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs, p_num_nbs_mar, a_fea, b_fea, d_anb, d_bnb, d_nbs, num_nbs_mar, affinity_labels,
            edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, d_bond_nei, p_nei_nbs, d_nei_nbs, p_nbs_mar, d_nbs_mar)


# Model initialization
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=min(1.0 / math.sqrt(m.weight.data.shape[-1]), 0.1))
        nn.init.constant_(m.bias, 0)


def loading_emb(filepath):

    with open(filepath, 'rb') as f:
        data_pack = pickle.load(f)

        pretrained_fea = data_pack[0]
        surface_fea = data_pack[1]
        p_edge_fea = data_pack[2]
        # p_adj = data_pack[2]
        p_anb = data_pack[3]
        p_bnb = data_pack[4]
        p_nbs = data_pack[5]
        p_num_nbs_mar = data_pack[6]

        a_fea = data_pack[7]
        b_fea = data_pack[8]
        d_anb = data_pack[9]
        d_bnb = data_pack[10]
        d_nbs = data_pack[11]
        num_nbs_mar = data_pack[12]

        affinity_labels = data_pack[13]

        edge_residue_to_atom = data_pack[14]
        edge_atom_to_residue = data_pack[15]
        p_nei = data_pack[16]
        d_nei = data_pack[17]
        p_bond_nei = data_pack[18]
        d_bond_nei = data_pack[19]
        p_nei_nbs = data_pack[20]
        d_nei_nbs = data_pack[21]
        p_nbs_mar = data_pack[22]
        d_nbs_mar = data_pack[23]

    return (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs, p_num_nbs_mar,
            a_fea, b_fea, d_anb, d_bnb, d_nbs, num_nbs_mar, affinity_labels,
            edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei,
            d_bond_nei, p_nei_nbs, d_nei_nbs, p_nbs_mar, d_nbs_mar)


def split_train_valid_test(train_data, valid_data, test_data):

    return (train_data, valid_data, test_data)