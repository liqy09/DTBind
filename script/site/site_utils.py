import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
import numpy as np



class ProteinDrugDataset(Dataset):
    def __init__(self, protein_folder, drug_folder, mapping_file, pdb_ids_file):
        self.protein_folder = protein_folder
        self.drug_folder = drug_folder
        self.mapping = self.load_mapping(mapping_file)
        self.pdb_ids = self.load_pdb_ids(pdb_ids_file)

        existing_protein_files = {os.path.splitext(file)[0] for file in os.listdir(self.protein_folder) if file.endswith(".pt")}
        existing_drug_files = {os.path.splitext(file)[0] for file in os.listdir(self.drug_folder) if file.endswith(".pt")}
        self.pdb_ids = [pdb_id for pdb_id in self.pdb_ids if pdb_id in existing_protein_files and self.mapping[pdb_id] in existing_drug_files]

    def load_mapping(self, mapping_file):
        mapping = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                pdb_id, drug_id = line.strip().split()
                mapping[pdb_id] = drug_id
        return mapping

    def load_pdb_ids(self, pdb_ids_file):
        with open(pdb_ids_file, 'r') as f:
            pdb_ids = [line.strip() for line in f]
        return pdb_ids

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        drug_id = self.mapping[pdb_id]

        protein_file = os.path.join(self.protein_folder, f"{pdb_id}.pt")
        protein_data = torch.load(protein_file)

        drug_file = os.path.join(self.drug_folder, f"{drug_id}.pt")
        drug_data = torch.load(drug_file)

        return protein_data, drug_data


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            inputs = torch.clamp(inputs, 1e-6, 1 - 1e-6)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

