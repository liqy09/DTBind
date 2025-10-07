import sys
import torch
from pyexpat.errors import messages
from torch import nn
from torch_cluster import knn_graph, radius_graph
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
import numpy as np
import math
import time
import pickle
import torch.optim as optim
from torch.nn.utils import weight_norm
# from torch.nn.utils.parametrizations import weight_norm
import logging
from aff_utils import *
import os
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'GNNpocket_depth': 3,
    'GNNvert_depth': 3,
    'GNNdti_depth': 3,
    'pairwise_depth': 3,
    'k_head': 2,
    'hidden_size1': 128,
    'hidden_size2': 128,
    'hidden_size3': 128,
    'hidden_size4': 128,
    'atom_fdim': 82,
    'bond_fdim': 6,
    'pro_pretrained_dim': 1024,
    'pro_surface_dim': 10,
    'pro_bond_fdim': 8,
    'dti_dis_thr': 6.0,
    'pro_dis_thr': 12.0,
    'edge_dim': 8,
    'h_dim': 64,
    'h_out': 64,
    'learning_rate': 0.0005,
    'weight_decay': 0,
    'step_size': 20,
    'gamma': 0.5,
    'n_epoch': 50,
    }



class PocketGNN(nn.Module):
    def __init__(self, pro_pretrained_dim, pro_surface_dim, pro_bond_fdim, hidden_size2, GNNpocket_depth, alpha=0.1):
        super(PocketGNN, self).__init__()
        self.hidden_size2 = hidden_size2
        self.pro_bond_fdim = pro_bond_fdim
        self.GNNpocket_depth = GNNpocket_depth
        self.pro_pretrained_dim = pro_pretrained_dim
        self.pro_surface_dim = pro_surface_dim
        self.alpha = alpha

        # Embedding layers for features
        self.pretrained_embedding = nn.Sequential(
            nn.Linear(pro_pretrained_dim, hidden_size2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_size2, hidden_size2),
            nn.LeakyReLU(0.1),
        )

        self.surface_embedding = nn.Sequential(
            nn.Linear(pro_surface_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32)
        )


        self.edge_emdedding = nn.Linear(8, 32)

        self.final_embedding = nn.Sequential(
            nn.Linear(32 + hidden_size2, hidden_size2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_size2, hidden_size2),
            nn.LeakyReLU(0.1),
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=0.1)

        # Create separate parameters for each layer (one for each depth of GNN)
        self.WQ = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for _ in range(GNNpocket_depth)])
        self.WK = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for _ in range(GNNpocket_depth)])
        self.Wgather = nn.ModuleList([nn.Linear(self.hidden_size2 + 8, self.hidden_size2) for _ in range(GNNpocket_depth)])
        self.Wg = nn.ModuleList([nn.Linear(self.hidden_size2 * 2, self.hidden_size2) for _ in range(GNNpocket_depth)])
        self.attention_module = nn.ModuleList([nn.Linear(8 + self.hidden_size2 * 2, self.hidden_size2 + 8) for _ in range(GNNpocket_depth)])
        self.W_res = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for _ in range(GNNpocket_depth)])
        self.Wga = nn.ModuleList([nn.Linear(self.hidden_size2 + 32, self.hidden_size2) for _ in range(GNNpocket_depth)])

    def forward(self, p_pretrained_fea, p_surface_fea, p_edge_fea, atom_nb, bond_nb, nbs_mask):

        batch_size, n_residues, _ = p_pretrained_fea.size()

        p_pretrained_fea = self.pretrained_embedding(p_pretrained_fea)  # [batch_size, n_residues, hidden_size2]
        p_surface_fea = self.surface_embedding(p_surface_fea)  # [batch_size, n_residues, 32]


        p_edge_fea = self.edge_emdedding(p_edge_fea)

        h = p_pretrained_fea
        initial_h = h.clone()
        # t = torch.zeros_like(h)

        for depth in range(self.GNNpocket_depth):
            t = F.relu(self.message_passing(h, p_edge_fea, atom_nb, bond_nb, nbs_mask, depth), 0.1)
            h_new = F.relu(self.Wg[depth](torch.cat([h, t], dim=-1)))
            h = h + h_new

        h = h + initial_h
        h = torch.cat([h, p_surface_fea], dim=-1)  # Concatenate along the feature dimension

        h = self.final_embedding(h)  # Embed to hidden_size2

        return h


    def message_passing(self, h, edge_fea, atom_nb, bond_nb, nbs_mask, depth):
        batch_size, n_res, feature_dim = h.size()
        n_nbs = atom_nb.size(2)
        nbs_mask = nbs_mask.view(batch_size, n_res, n_nbs, 1)
        nei_tensor = atom_nb.view(batch_size, n_res * n_nbs)
        vertex_nei = h.gather(1, nei_tensor.unsqueeze(-1).expand(-1, -1,
                                                                 feature_dim))  # [batch_size, n_res * n_nbs, feature_dim]

        nei = vertex_nei.view(batch_size, n_res, n_nbs, -1)  # [batch_size, n_res, n_nbs, feature_dim + edge_feature_dim]

        edge_initial = edge_fea.unsqueeze(2)  # [batch_size, n_vertex, 1, 32]
        edge_nei = torch.gather(edge_initial.expand(-1, -1, n_nbs, -1), 1,
                                bond_nb.view(batch_size, n_res, n_nbs, 1).expand(-1, -1, -1,
                                                                                     32))  # [batch_size, n_vertex, n_nbs, 6]
        edge_nei = edge_nei.view(batch_size, n_res, n_nbs, 32)

        l_nei = torch.cat((edge_nei, nei), -1)

        # Aggregate weighted neighbor features
        ver_nei = F.relu(self.Wga[depth](l_nei))  # [batch_size, n_res, n_nbs, output_dim]
        # Aggregate messages for each residue by summing over the neighbors
        messages_i = torch.sum(ver_nei * nbs_mask, dim=2)  # Sum over neighbors: [batch_size, n_res, output_dim]

        return messages_i



class LigandGNN(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden_size1, k_head, GNNvert_depth):
        super(LigandGNN, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size1 = hidden_size1
        self.k_head = k_head
        self.GNNvert_depth = GNNvert_depth

        self.vertex_embedding = nn.Linear(self.atom_fdim, self.hidden_size1)

        self.W_a_main = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.k_head)])
             for _ in range(self.GNNvert_depth)]
        )
        self.W_a_super = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.k_head)])
             for _ in range(self.GNNvert_depth)]
        )
        self.W_main = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.k_head)])
             for _ in range(self.GNNvert_depth)])

        self.W_bmm = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.hidden_size1, 1) for _ in range(self.k_head)])
             for _ in range(self.GNNvert_depth)]
        )
        self.W_super = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GNNvert_depth)]
        )

        self.W_main_to_super = nn.ModuleList(
            [nn.Linear(self.hidden_size1 * self.k_head, self.hidden_size1) for _ in range(self.GNNvert_depth)]
        )
        self.W_super_to_main = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GNNvert_depth)]
        )
        self.W_zm1 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GNNvert_depth)])
        self.W_zm2 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GNNvert_depth)])
        self.W_zs1 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GNNvert_depth)])
        self.W_zs2 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for _ in range(self.GNNvert_depth)])

        self.GRU_main = nn.GRU(self.hidden_size1, self.hidden_size1, batch_first=True)
        self.GRU_super = nn.GRU(self.hidden_size1, self.hidden_size1, batch_first=True)

        # WLN parameters
        self.label_U2 = nn.ModuleList(
            [nn.Linear(self.hidden_size1 + self.bond_fdim, self.hidden_size1) for _ in range(self.GNNvert_depth)]
        )
        self.label_U1 = nn.ModuleList(
            [nn.Linear(self.hidden_size1 * 2, self.hidden_size1) for _ in range(self.GNNvert_depth)]
        )

    def GraphConv_module(self, batch_size, atom_fea, bond_fea, d_anb, d_bnb, d_nbs_mask, d_mask):

        n_vertex = atom_fea.size(1)  # batch_size x n_vertex x fdim
        vertex_initial = atom_fea
        edge_initial = bond_fea
        # print(d_mask.shape)  # [32, 185]

        # Embedding the vertices (nodes)  [32, 185, 128]
        # vertex_feature = F.leaky_relu(self.vertex_embedding(vertex_initial), 0.1)
        vertex_feature = F.leaky_relu(self.vertex_embedding(vertex_initial))
        # edge_initial = self.bond_embedding(edge_initial)

        super_feature = torch.sum(vertex_feature * d_mask.view(vertex_feature.size(0), vertex_feature.size(1), 1), dim=1, keepdim=True)
        # print(super_feature.shape)

        for GWM_iter in range(self.GNNvert_depth):
            vertex_heads = []
            for k in range(self.k_head):
                a_main = torch.tanh(self.W_a_main[GWM_iter][k](vertex_feature))
                a_super = torch.tanh(self.W_a_super[GWM_iter][k](super_feature))

                a = self.W_bmm[GWM_iter][k](a_main * a_super)

                attn = mask_softmax(a.view(vertex_feature.size(0), -1), d_mask).view(vertex_feature.size(0), -1, 1)

                k_main_to_super = torch.bmm(attn.transpose(1, 2), self.W_main[GWM_iter][k](vertex_feature))

                if k == 0:
                    m_main_to_super = k_main_to_super
                else:
                    m_main_to_super = torch.cat([m_main_to_super, k_main_to_super], dim=-1)

            main_to_super = torch.tanh(self.W_main_to_super[GWM_iter](m_main_to_super))

            main_self = self.wln_unit(vertex_feature, edge_initial, d_anb, d_bnb, d_nbs_mask, GWM_iter)
            super_to_main = torch.tanh(self.W_super_to_main[GWM_iter](super_feature))
            super_self = torch.tanh(self.W_super[GWM_iter](super_feature))

            z_main = torch.sigmoid(self.W_zm1[GWM_iter](main_self) + self.W_zm2[GWM_iter](super_to_main))
            hidden_main = (1 - z_main) * main_self + z_main * super_to_main
            vertex_feature = hidden_main

            z_supper = torch.sigmoid(self.W_zs1[GWM_iter](super_self) + self.W_zs2[GWM_iter](main_to_super))
            hidden_super = (1 - z_supper) * super_self + z_supper * main_to_super
            super_feature = hidden_super

            super_feature = super_feature.sum(dim=1, keepdim=True)

        return vertex_feature, super_feature


    def wln_unit(self, vertex_feature, edge_initial, atom_adj, bond_adj, d_nbs_mask, GWM_iter):
        n_vertex = vertex_feature.size(1)
        n_nbs = d_nbs_mask.size(2)
        batch_size = atom_adj.size(0)

        # vertex_mask = d_mask.view(batch_size, n_vertex, 1)
        nbs_mask = d_nbs_mask.view(batch_size, n_vertex, n_nbs, 1)

        atom_adj = atom_adj.view(batch_size, n_vertex, n_nbs, 1)  # 增加一个维度，方便后续扩展

        atom_adj = atom_adj.expand(-1, -1, -1,
                                   vertex_feature.size(2))

        vertex_nei = torch.gather(vertex_feature.view(batch_size, n_vertex, 1, -1).expand(-1, -1, n_nbs, -1), 1,
                                  atom_adj).view(batch_size, n_vertex, n_nbs, self.hidden_size1)

        edge_initial = edge_initial.unsqueeze(2)  # [batch_size, n_vertex, 1, 6]
        edge_nei = torch.gather(edge_initial.expand(-1, -1, n_nbs, -1), 1,
                                bond_adj.view(batch_size, n_vertex, n_nbs, 1).expand(-1, -1, -1, 6))  # [batch_size, n_vertex, n_nbs, 6]

        edge_nei = edge_nei.view(batch_size, n_vertex, n_nbs, 6)

        l_nei = torch.cat((vertex_nei, edge_nei), -1)
        nei_label = F.leaky_relu(self.label_U2[GWM_iter](l_nei), 0.1)
        nei_label = torch.sum(nei_label * nbs_mask, dim=-2)
        new_label = torch.cat((vertex_feature, nei_label), 2)
        new_label = self.label_U1[GWM_iter](new_label)
        vertex_features = F.leaky_relu(new_label, 0.1)

        return vertex_features


class HeteroGraphConv(nn.Module):
    def __init__(self, hidden_size4, edge_dim, update_dim, GNNdti_depth):
        super(HeteroGraphConv, self).__init__()
        self.hidden_size4 = hidden_size4
        self.edge_dim = edge_dim
        self.update_dim = update_dim
        self.GNNdti_depth = GNNdti_depth

        self.Wu = nn.Linear(hidden_size4, update_dim)

        self.Wgather = nn.Linear(hidden_size4 + 32, hidden_size4)

        self.WQ_d = nn.ModuleList(
            [nn.Linear(hidden_size4, hidden_size4) for _ in range(self.GNNdti_depth)]
        )
        self.Wdc = nn.ModuleList(
            [nn.Linear(hidden_size4 + 32, hidden_size4) for _ in range(self.GNNdti_depth)]
        )
        self.WK_d = nn.ModuleList(
            [nn.Linear(hidden_size4, hidden_size4) for _ in range(self.GNNdti_depth)]
        )

        self.pb_emb = nn.Linear(8, 32)
        self.db_emb = nn.Linear(8, 32)

        self.WQ_p = nn.ModuleList(
            [nn.Linear(hidden_size4, hidden_size4) for _ in range(self.GNNdti_depth)]
        )
        self.Wpc = nn.ModuleList(
            [nn.Linear(hidden_size4 + 32, hidden_size4) for _ in range(self.GNNdti_depth)]
        )
        self.WK_p = nn.ModuleList(
            [nn.Linear(hidden_size4, hidden_size4) for _ in range(self.GNNdti_depth)]
        )

        self.W_p = nn.ModuleList(
            [nn.Linear(hidden_size4 + 32, hidden_size4) for _ in range(self.GNNdti_depth)]
        )
        self.W_d = nn.ModuleList(
            [nn.Linear(hidden_size4 + 32, hidden_size4) for _ in range(self.GNNdti_depth)]
        )

        self.Wd_up = nn.ModuleList(
            [nn.Linear(hidden_size4, update_dim) for _ in range(self.GNNdti_depth)]
        )
        self.Wp_up = nn.ModuleList(
            [nn.Linear(hidden_size4, update_dim) for _ in range(self.GNNdti_depth)]
        )

        self.Wu_drug = nn.Sequential(
            nn.Linear(hidden_size4, hidden_size4),
            nn.ReLU(),
            nn.Linear(hidden_size4, hidden_size4)
        )
        self.Wu_protein = nn.Sequential(
            nn.Linear(hidden_size4, hidden_size4),
            nn.ReLU(),
            nn.Linear(hidden_size4, hidden_size4)
        )

    def forward(self, protein_features, drug_features, edge_features_residue_to_atom, edge_features_atom_to_residue,
                protein_neighbors, drug_neighbors, protein_bonds, drug_bonds, p_hetero_mask, d_hetero_mask):


        drug_features = self.Wu_drug(drug_features)  # h_d^0
        protein_features = self.Wu_protein(protein_features)  # h_p^0

        edge_features_residue_to_atom = self.db_emb(edge_features_residue_to_atom)
        edge_features_atom_to_residue = self.pb_emb(edge_features_atom_to_residue)

        updated_drug = drug_features
        updated_protein = protein_features
        # Iterate through the layers of the GNN
        for iter in range(self.GNNdti_depth):
            updated_drug_1, updated_protein_1 = self.GNN(updated_protein, updated_drug, edge_features_residue_to_atom, edge_features_atom_to_residue,
                protein_neighbors, drug_neighbors, protein_bonds, drug_bonds, p_hetero_mask, d_hetero_mask, iter)
            updated_drug = updated_drug_1
            updated_protein = updated_protein_1

        updated_drug = updated_drug + drug_features
        updated_protein = updated_protein + protein_features


        return updated_protein, updated_drug

    def GNN(self, protein_features, drug_features, edge_features_residue_to_atom, edge_features_atom_to_residue,
                protein_neighbors, drug_neighbors, protein_bonds, drug_bonds, p_hetero_mask, d_hetero_mask, iter):


        n_d_nodes = drug_features.size(1)  # Number of drug nodes
        # print(f"n_d_nodes:{n_d_nodes}")
        n_p_nodes = protein_features.size(1)  # Number of protein nodes
        # print(f"n_p_nodes:{n_p_nodes}")

        batch_size = drug_features.size(0)  # Batch size
        d_nbs = drug_neighbors.size(2)  # Number of neighbors for each drug node
        # print(f"d_nbs:{d_nbs}")
        p_nbs = protein_neighbors.size(2)  # Number of neighbors for each protein node
        # print(f"p_nbs:{p_nbs}")

        # Reshape the hetero mask for drugs (to match neighbor dimensions)
        d_mask = d_hetero_mask.view(batch_size, n_d_nodes, d_nbs, 1)  # [batch_size, n_d_nodes, n_nbs, 1]
        # print(f"d_mask:{d_mask}")
        p_mask = p_hetero_mask.view(batch_size, n_p_nodes, p_nbs, 1)  # [batch_size, n_p_nodes, n_nbs, 1]
        # print(f"p_mask:{p_mask}")

        drug_neighbors_expanded = drug_neighbors.view(batch_size, n_d_nodes, d_nbs, 1)  # Precompute once
        # print(f"drug_neighbors_expanded:{drug_neighbors_expanded}")

        drug_neighbors_expanded = drug_neighbors_expanded.expand(-1, -1, -1, protein_features.size(
            2))  # [batch_size, n_d_nodes, n_nbs, feature_dim]

        d_edge = edge_features_residue_to_atom.unsqueeze(2)  # [batch_size, n_d_nodes, 1, edge_feature_dim]
        # print(f"d_edge:{d_edge}")

        protein_nei_expanded = protein_neighbors.view(batch_size, n_p_nodes, p_nbs,
                                                      1)  # [batch_size, n_d_nodes, n_nbs, 1]
        # print(f"protein_nei_expanded:{protein_nei_expanded}")
        protein_neighbors_expanded = protein_nei_expanded.expand(-1, -1, -1, drug_features.size(
            2))  # [batch_size, n_p_nodes, n_nbs, feature_dim]

        p_edge = edge_features_atom_to_residue.unsqueeze(2)  # [batch_size, n_d_nodes, 1, edge_feature_dim]

        drug_neighbor_features = torch.gather(
            protein_features.view(batch_size, n_p_nodes, 1, -1).expand(-1, -1, n_d_nodes, -1),
            1, drug_neighbors_expanded).view(batch_size, n_d_nodes, d_nbs, -1)
        # print(f"drug_neighbor_feature:{drug_neighbor_features}")

        # Edge features for residue-to-atom connections
        d_edge_nei = torch.gather(d_edge.expand(-1, -1, n_d_nodes, -1), 1,
                                  drug_bonds.view(batch_size, n_d_nodes, d_nbs, 1).expand(-1, -1, -1,
                                                                                          32))  # [batch_size, n_d_nodes, n_nbs, edge_feature_dim]

        # Calculate the attention scores
        Q_d = self.WQ_d[iter](drug_features)  # Query for drug nodes
        Q_d = Q_d.unsqueeze(2)  # Shape: [batch_size, num_drug_nodes, 1, feature_dim]

        # Concatenate drug edge features and neighbors' features
        drug_neighbors_combined = torch.cat([d_edge_nei, drug_neighbor_features],
                                            dim=-1)  # [batch_size, n_d_nodes, n_nbs, feature_dim]
        drug_neighbors_transformed = self.Wdc[iter](
            drug_neighbors_combined)  # [batch_size, n_d_nodes, n_nbs, feature_dim]

        # Key for drug neighbors
        K_d = self.WK_d[iter](drug_neighbors_transformed)  # [batch_size, n_d_nodes, n_nbs, feature_dim]
        attention_scores = torch.matmul(Q_d, K_d.transpose(2, 3))  # [batch_size, n_d_nodes, 1, n_nbs]
        attention_scores = attention_scores / (Q_d.size(-1) ** 0.5)  # Scale by the feature dimension size
        attention_scores = attention_scores.transpose(2, 3)  # [batch_size, n_d_nodes, n_nbs, 1]


        alpha_d = mask_softmax(attention_scores, d_mask, dim=-2)

        drug_neighbors_features = torch.cat([d_edge_nei, drug_neighbors_transformed], dim=-1)
        drug_neighbors_features = F.leaky_relu(self.W_d[iter](drug_neighbors_features), 0.1)

        # Expand alpha_d to match the neighbor features dimensions
        alpha_d_expanded = alpha_d.expand(-1, -1, -1, drug_neighbors_features.size(-1))
        weighted_drug_neighbors = torch.sum(alpha_d_expanded * drug_neighbors_features, dim=2, keepdim=True)

        # Update drug node features with residual connection
        drug_neighbors = weighted_drug_neighbors.squeeze(2)
        updated_drug_feature = F.leaky_relu(self.Wd_up[iter](drug_features + drug_neighbors))

        # Process protein nodes
        protein_nei = torch.gather(
            drug_features.view(batch_size, n_d_nodes, 1, -1).expand(-1, -1, n_p_nodes, -1),
            1, protein_neighbors_expanded).view(batch_size, n_p_nodes, p_nbs, -1)

        # Edge features for residue-to-atom connections
        p_edge_nei = torch.gather(p_edge.expand(-1, -1, n_p_nodes, -1), 1,
                                  protein_bonds.view(batch_size, n_p_nodes, d_nbs, 1).expand(-1, -1, -1,
                                                                                             32))  # [batch_size, n_d_nodes, n_nbs, edge_feature_dim]

        # Attention mechanism for protein nodes
        Q_p = self.WQ_p[iter](protein_features)
        Q_p = Q_p.unsqueeze(2)

        # Concatenate protein edge features and drug neighbors' features
        protein_neighbors_features_combined = torch.cat([p_edge_nei, protein_nei], dim=-1)
        protein_neighbors_transformed = self.Wpc[iter](protein_neighbors_features_combined)

        # Key for protein neighbors
        K_p = self.WK_p[iter](protein_neighbors_transformed)  # [batch_size, n_p_nodes, n_nbs, feature_dim]
        attention_scores_p = torch.matmul(Q_p, K_p.transpose(2, 3))  # [batch_size, n_p_nodes, 1, n_nbs]
        attention_scores_p = attention_scores_p / (Q_p.size(-1) ** 0.5)  # Scale by the feature dimension size
        attention_scores_p = attention_scores_p.transpose(2, 3)  # [batch_size, n_p_nodes, n_nbs, 1]

        # Apply mask for protein nodes
        alpha_p = mask_softmax(attention_scores_p, p_mask, dim=-2)

        # Concatenate the protein neighbors' edge and feature information for weighted sum
        protein_neighbors_features = torch.cat([p_edge_nei, protein_neighbors_transformed], dim=-1)
        protein_neighbors_features = F.leaky_relu(self.W_p[iter](protein_neighbors_features), 0.1)

        # Expand alpha_p to match the neighbor features dimensions
        alpha_p_expanded = alpha_p.expand(-1, -1, -1, protein_neighbors_features.size(-1))
        weighted_protein_neighbors = torch.sum(alpha_p_expanded * protein_neighbors_features, dim=2, keepdim=True)

        # Update protein node features with residual connection
        protein_neighbors = weighted_protein_neighbors.squeeze(2)
        updated_protein_feature = F.leaky_relu(self.Wp_up[iter](protein_features + protein_neighbors))

        return updated_drug_feature, updated_protein_feature


class Affinity(nn.Module):
    def __init__(self, hidden_size, output_size, pairwise_depth):
        super(Affinity, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pairwise_depth = pairwise_depth

        self.d = nn.Linear(self.hidden_size, self.hidden_size)
            # nn.Linear(self.hidden_size, self.hidden_size)
        self.p = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_v = nn.Linear(hidden_size * 2, hidden_size)
        self.att = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(pairwise_depth)])
        self.Wv = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(pairwise_depth)])
        self.Wu = nn.Linear(hidden_size * 2, hidden_size)

        self.W1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)


        self.W2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        self.W3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1)
        )

        self.Wu1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.2),
        )

        self.W5 = nn.Linear(hidden_size, 1)
        

    def forward(self, protein_features, drug_features, pu_mask, du_mask):

        batch_size = protein_features.size(0)
        n_protein = protein_features.size(1)
        n_drug = drug_features.size(1)

        # 特征变换
        drug_feat = F.leaky_relu(self.d(drug_features), 0.1) * du_mask.unsqueeze(-1)
        prot_feat = F.leaky_relu(self.p(protein_features), 0.1) * pu_mask.unsqueeze(-1)

        # 全局表示
        u_d = drug_feat.sum(dim=1)
        u_p = prot_feat.sum(dim=1)

        prot_exp = prot_feat.unsqueeze(2)  # [B, P, 1, H]
        drug_exp = drug_feat.unsqueeze(1)  # [B, 1, D, H]
        mij = torch.cat([
            prot_exp.expand(-1, -1, n_drug, -1),
            drug_exp.expand(-1, n_protein, -1, -1)
        ], dim=-1)

        mij = mij.view(batch_size, -1, 2 * self.hidden_size)  # [B, P*D, 2H]
        mij = F.leaky_relu(self.W_v(mij), 0.1)  # 初始变换

        #计算 m_u = W_u[u_d, u_p]
        for i in range(self.pairwise_depth):

            u_combined = torch.cat([u_d, u_p], dim=-1)  # (hidden_size * 2,)
            m_u = F.leaky_relu(self.Wu(u_combined))  # (hidden_size,)

            mij_i = self.Wv[i](mij)
            alpha_ij_i = self.att[i](mij_i)
            alpha_ij_i = alpha_ij_i.expand(-1, -1,
                                           self.hidden_size)  # (batch_size, batch_size_protein * batch_size_drug, hidden_size)
            global_i = alpha_ij_i * mij_i  # (batch_size, batch_size_protein * batch_size_drug, hidden_size)
            mij = global_i + mij

        global_i = global_i + mij
        global1 = self.W1(global_i)

        global1 = torch.sum(global1, dim=1)
        global2 = self.W2(global1)
        global3 = self.W3(global2)

        # 结合全局信息与 m_u
        mu = self.Wu1(torch.cat([m_u, global3], dim=-1))

        output = self.W5(mu)
        return output

class DTIModel(nn.Module):
    def __init__(self, params):
        super(DTIModel, self).__init__()
        self.params = params

        self.ligand_gnn = LigandGNN(
            atom_fdim=params['atom_fdim'],
            bond_fdim=params['bond_fdim'],
            hidden_size1=params['hidden_size1'],
            k_head=params['k_head'],
            GNNvert_depth=params['GNNvert_depth']
        )

        self.pocket_gnn = PocketGNN(
            pro_pretrained_dim=params['pro_pretrained_dim'],
            pro_surface_dim=params['pro_surface_dim'],
            pro_bond_fdim=params['pro_bond_fdim'],
            hidden_size2=params['hidden_size2'],
            GNNpocket_depth=params['GNNpocket_depth']
        )

        self.hetero_graph_conv = HeteroGraphConv(
            hidden_size4=params['hidden_size4'],        # hidden_dim
            edge_dim=params['hidden_size4'] * 2 + 8,    # edge_dim
            update_dim=params['hidden_size4'],          # update_dim
            GNNdti_depth=params['GNNdti_depth']
        )

        self.affinity_model = Affinity(
            hidden_size=params['hidden_size3'],
            output_size=1,
            pairwise_depth=params['pairwise_depth']
        )

    def forward(self, pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs_mask, p_mask, a_fea, b_fea, d_anb,
                d_bnb, num_nbs_mar, d_mask, edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, d_bond_nei,
                p_hetero_mask, d_hetero_mask):

        batch_size = 32

        updated_drug_features, super_feature = self.ligand_gnn.GraphConv_module(
            batch_size=batch_size,
            atom_fea=a_fea,
            bond_fea=b_fea,
            d_anb=d_anb,
            d_bnb=d_bnb,
            d_nbs_mask=num_nbs_mar,
            d_mask=d_mask,
        )

        updated_protein_features = self.pocket_gnn(
            p_pretrained_fea=pretrained_fea,
            p_surface_fea=surface_fea,
            p_edge_fea=p_edge_fea,
            atom_nb=p_anb,
            bond_nb=p_bnb,
            nbs_mask=p_nbs_mask,
        )

        updated_protein_features, updated_drug_features = self.hetero_graph_conv(
            updated_protein_features,
            updated_drug_features,
            edge_residue_to_atom,
            edge_atom_to_residue,
            p_nei,
            d_nei,
            p_bond_nei,
            d_bond_nei,
            p_hetero_mask,
            d_hetero_mask
        )

        affinity_pred = self.affinity_model(
            protein_features=updated_protein_features,
            drug_features=updated_drug_features,
            pu_mask=p_mask,
            du_mask=d_mask
        )

        return affinity_pred


def softmax(a, dim=-1):
    a_max = torch.max(a, dim, keepdim=True)[0]
    a_exp = torch.exp(a - a_max)
    a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
    return a_softmax


def mask_softmax(a, mask, dim=-1):
    a_max = torch.max(a, dim, keepdim=True)[0]
    a_exp = torch.exp(a - a_max)
    a_exp = a_exp * mask
    a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
    return a_softmax

