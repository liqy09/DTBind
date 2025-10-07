import os
import torch
import csv
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data

# 定义药物特征维度
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
             'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
             'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U',
             'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1  # 元素符号特征 + 原子度数、显性价、隐性价、芳香性
bond_fdim = 6
max_nb = 6  # 最大邻居数

# 用于one-hot编码原子和键的特征
def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)

# 转换为PyTorch Geometric格式的图
def Mol2Graph(smiles, drug_id, output_folder, max_nb=6):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string for {drug_id}. Skipping...")
        return

    # Remove hydrogen atoms
    mol = Chem.RemoveHs(mol)

    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # 初始化矩阵
    atom_feature_matrix = np.zeros((n_atoms, atom_fdim), dtype=np.float32)
    bond_feature_matrix = np.zeros((n_bonds, bond_fdim), dtype=np.float32)
    atom_nb = np.zeros((n_atoms, max_nb), dtype=int)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=int)
    num_nbs = np.zeros((n_atoms,), dtype=int)
    adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)

    # 提取原子特征
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_feature_matrix[idx] = atom_features(atom)

    # 提取键特征并建立邻接信息
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bond_feat = bond_features(bond)
        bond_feature_matrix[idx] = bond_feat

        # 如果邻居数未达到限制，则添加
        if num_nbs[a1] < max_nb:
            atom_nb[a1, num_nbs[a1]] = a2
            bond_nb[a1, num_nbs[a1]] = idx
            num_nbs[a1] += 1
        if num_nbs[a2] < max_nb:
            atom_nb[a2, num_nbs[a2]] = a1
            bond_nb[a2, num_nbs[a2]] = idx
            num_nbs[a2] += 1

        adj_matrix[a1, a2] = 1
        adj_matrix[a2, a1] = 1

    node_features = torch.tensor(atom_feature_matrix, dtype=torch.float32)  # 原子特征
    edge_index = []
    edge_attr = []


    for i in range(len(bond_nb)):
        for j in range(num_nbs[i]):
            if bond_nb[i, j] != 0:
                edge_index.append([i, atom_nb[i, j]])  # 原子i和原子atom_nb[i, j]之间有一条边
                edge_attr.append(bond_feature_matrix[bond_nb[i, j]])  # 对应的边特征

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # PyG要求的格式是[2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    output_file_path = os.path.join(output_folder, f'{drug_id}.pt')
    torch.save(data, output_file_path)
    print(f"Saved graph for {drug_id} to {output_file_path}")

def process_and_save_graphs_from_tsv(input_tsv_file, output_folder, max_nb=6):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_tsv_file, mode='r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')

        print(f"列名: {reader.fieldnames}")
        for row in reader:
            drug_id = row['DrugBank ID']
            smiles = row['SMILES']
            Mol2Graph(smiles, drug_id, output_folder, max_nb=max_nb)

input_tsv_file = './Data/dti/drug_smiles.tsv'
output_folder = './Data/dti/drug_graph'

process_and_save_graphs_from_tsv(input_tsv_file, output_folder, max_nb=6)
