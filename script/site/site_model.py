import sys
import torch
from torch import nn
from torch_cluster import knn_graph, radius_graph
import torch.nn.functional as F
# import torch_scatter
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_softmax
from torch_geometric.utils import scatter
import numpy as np
from pyexpat.errors import messages
import math
import time
import pickle
import torch.optim as optim
from torch.nn.utils import weight_norm
import logging
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from site_utils import *
import os


torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'proGNN_depth': 3,
    'drugGNN_depth': 3,
    'crossatt_depth': 3,
    'hidden_size1': 128,
    'hidden_size2': 128,
    'hidden_size3': 128,
    'hidden_size4': 128,
    'crossatt_head': 4,
    'proselfatt_head': 4,
    'pro_pretrained_dim': 1024,
    'pro_surface_dim': 10,
    'pro_bond_fdim': 8,
    'edge_dim': 8,
    'drugfea_dim': 82,
    'drug_edge_dim': 6,
    'dropout': 0.1,
    }



class ProteinGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, hidden_dim, lambda_=0.5):
        super(ProteinGraphConv, self).__init__(aggr='add')
        self.lambda_ = lambda_
        self.d_k = hidden_dim

        # 线性变换
        # self.lin_node = nn.Linear(in_channels, hidden_dim)
        self.lin_node_out = nn.Linear(hidden_dim, out_channels)
        self.lin_edge = nn.Linear(edge_dim, hidden_dim)

        # 注意力权重变换
        self.W_Q_node = nn.Linear(hidden_dim, hidden_dim)
        self.W_K_node = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W_K_edge = nn.Linear(hidden_dim, hidden_dim)

        # 聚合权重
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_e = nn.Linear(hidden_dim, hidden_dim)  # 新增边特征的聚合权重

    def forward(self, x, edge_index, edge_attr, batch):
        # 节点特征和边特征的线性变换
        # x = F.relu(self.lin_node(x))
        edge_attr = self.lin_edge(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, batch=batch)

    def message(self, x_i, x_j, edge_attr, batch, edge_index):

        Q_v = self.W_Q_node(x_j)
        K_v = self.W_K_node(x_i)
        alpha_v = torch.sum(Q_v * K_v, dim=-1, keepdim=True) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))


        Q_e = self.W_Q_edge(edge_attr)
        K_e = self.W_K_edge(edge_attr)
        alpha_e = torch.sum(Q_e * K_e, dim=-1, keepdim=True) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # 批内softmax归一化 (关键改进点)
        alpha_v = self._batch_softmax(alpha_v, batch, edge_index)  # [E, 1]
        alpha_e = self._batch_softmax(alpha_e, batch, edge_index)  # [E, 1]

        # 注意力融合
        alpha_ij = self.lambda_ * alpha_v + (1 - self.lambda_) * alpha_e

        V = self.W_v(x_j)
        E = self.W_e(edge_attr)  # 对边特征进行线性变换
        return alpha_ij * (V + E)  # [E, d]

    def _batch_softmax(self, alpha, batch, edge_index):
        node_indices = edge_index[0]
        batch_indices = batch[node_indices]

        alpha = scatter_softmax(alpha.squeeze(-1), batch_indices, dim=0).unsqueeze(-1)
        return alpha

    def update(self, aggr_out, x):
        return F.relu(self.lin_node_out(x) + aggr_out)



class DrugGNN(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(DrugGNN, self).__init__(aggr='add')
        # WLN parameters
        self.label_U2 = nn.Sequential(  # assume no edge feature transformation
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1),
        )
        self.label_U1 = nn.Linear(out_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr=None):
        if edge_attr is None:
            z = x_j
        else:
            z = torch.cat([x_j, edge_attr], dim=-1)
        return self.label_U2(z)

    def update(self, message, x):
        z = torch.cat([x, message], dim=-1)
        return self.label_U1(z)


class SelfAttentionBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, n, d]"""
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(x + attn_output)
        return output


class GatedFusion(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, H_c, H_s):

        combined = torch.cat([H_c, H_s], dim=-1)  # [B, n, 2d]
        g = self.gate(combined)  # [B, n, d]
        H_f = g * H_c + (1 - g) * H_s
        return H_f


class PredictionHead(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 1)
        )
        self.residual = nn.Linear(d_model, 1)

    def forward(self, x):
        """x: [B, n, d]"""
        residual = self.residual(x)  # [B, n, 1]
        mlp_out = self.mlp(x)  # [B, n, 1]
        # out = F.softmax((mlp_out + residual), dim=-1)
        out = torch.sigmoid(residual + mlp_out)
        # out = torch.nan_to_num(out)
        return out


class MultiHeadCrossAttention(nn.Module):


    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _create_batch_mask(self, batch_p, batch_l):

        mask = (batch_p.unsqueeze(-1) == batch_l.unsqueeze(0))  # [n_p, n_l]
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, n_p, n_l]

    def forward(self, protein, ligand, batch_protein, batch_ligand):

        Q = self.W_q(protein)  # [total_p, d]
        K = self.W_k(ligand)  # [total_l, d]
        V = self.W_v(ligand)  # [total_l, d]

        Q = Q.view(-1, self.num_heads, self.d_k).transpose(0, 1)  # [h, total_p, d_k]
        K = K.view(-1, self.num_heads, self.d_k).permute(1, 2, 0)  # [h, d_k, total_l]
        V = V.view(-1, self.num_heads, self.d_k).transpose(0, 1)  # [h, total_l, d_k]

        batch_mask = self._create_batch_mask(batch_protein, batch_ligand)  # [1, 1, total_p, total_l]

        scores = torch.matmul(Q, K) / (self.d_k ** 0.5)  # [h, total_p, total_l]
        scores = scores.masked_fill(~batch_mask, -1e9)  # 应用掩码

        attn_weights = F.softmax(scores, dim=-1)  # [h, total_p, total_l]
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # [h, total_p, d_k]
        context = context.transpose(0, 1).contiguous().view(-1, self.d_model)  # [total_p, d]

        output = self.W_o(context)
        output = self.layer_norm(protein + output)

        return output, attn_weights


class DTISite(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.prognn_depth = params['proGNN_depth']
        self.druggnn_depth = params['drugGNN_depth']
        self.crossatt_depth = params['crossatt_depth']
        self.hidden_size1 = params['hidden_size1']
        self.hidden_size2 = params['hidden_size2']
        self.hidden_size3 = params['hidden_size3']
        self.hidden_size4 = params['hidden_size4']
        self.crossatt_head = params['crossatt_head']
        self.proselfatt_head = params['proselfatt_head']
        self.pro_pretrained_dim = params['pro_pretrained_dim']
        self.pro_surface_dim = params['pro_surface_dim']
        self.pro_bond_fdim = params['pro_bond_fdim']
        self.pro_edge_dim = params['edge_dim']
        self.drugfea_dim = params['drugfea_dim']
        self.drug_edge_dim = params['drug_edge_dim']
        self.dropout = nn.Dropout(params['dropout'])

        self.drug_embedding = nn.Sequential(
            nn.Linear(self.drugfea_dim, self.hidden_size1),
            nn.LeakyReLU(0.1),
        )

        self.protein_embedding = nn.Sequential(
            nn.Linear(self.pro_pretrained_dim, self.hidden_size2),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_size2, self.hidden_size2),
        )

        self.protein_surf_embedding = nn.Sequential(
            nn.Linear(self.pro_surface_dim, self.hidden_size2 // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_size2 // 4, self.hidden_size2 // 4),
        )

        self.protein_fea_cat = nn.Sequential(
            nn.Linear(self.hidden_size2 + self.hidden_size2 // 4, self.hidden_size2),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_size2, self.hidden_size2),
        )


        self.druggnn = nn.ModuleList()
        for i in range(self.druggnn_depth):
            self.druggnn.append(DrugGNN(self.hidden_size1 + self.drug_edge_dim, self.hidden_size1))


        self.proteingnn = nn.ModuleList()
        for i in range(self.prognn_depth):
            self.proteingnn.append(
                ProteinGraphConv(self.hidden_size2, self.hidden_size2, self.pro_edge_dim, self.hidden_size2))

        self.cross_attention = MultiHeadCrossAttention(
            d_model=self.hidden_size3,
            num_heads=self.crossatt_head
        )

        self.proteinselfatt = SelfAttentionBlock(
            d_model=self.hidden_size3,
            num_heads=self.proselfatt_head
        )

        self.gated_fusion = GatedFusion(d_model=self.hidden_size4)

        self.pred_head = PredictionHead(d_model=self.hidden_size4)


    def forward(self, protein_data, drug_data):

        residue_initial = protein_data.x

        surf_initial = protein_data.surface_x
        # surf_initial = protein_data.surf_x

        surf_fea = self.protein_surf_embedding(surf_initial)

        residue_feature = self.protein_embedding(residue_initial)

        protein_feat = residue_feature
        protein_initial = protein_feat
        for i in range(self.prognn_depth):
            protein_feat = self.proteingnn[i](
                protein_feat,
                protein_data.edge_index,
                protein_data.edge_attr,
                protein_data.batch
            ) + protein_feat  # Residual connection
        protein_feat = protein_feat + protein_initial  # Add initial input at the end


        protein_feat = self.protein_fea_cat(torch.cat([protein_feat, surf_fea], dim=1))

        atom_initial = drug_data.x
        atom_feature = self.drug_embedding(atom_initial)
        # Drug feature extraction
        drug_feat = atom_feature
        drug_initial = drug_feat
        for i in range(self.druggnn_depth):
            drug_feat = self.druggnn[i](
                drug_feat,
                drug_data.edge_index,
                drug_data.edge_attr
            ) + drug_feat  # Residual connection
        drug_feat = drug_feat + drug_initial  # Add initial input at the end

        H_c, attn_weights = self.cross_attention(
            protein_feat,
            drug_feat,
            protein_data.batch,
            drug_data.batch
        )

        H_s = self.proteinselfatt(H_c)

        H_f = self.gated_fusion(H_c, H_s)


        preds = self.pred_head(H_f)


        return preds



