import os
import torch
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data


elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
             'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
             'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U',
             'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1  # 元素符号特征 + 原子度数、显性价、隐性价、芳香性
bond_fdim = 6
max_nb = 6  # 最大邻居数


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


def extract_3d_coordinates(mol):
    conformer = mol.GetConformer()
    atom_coords = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx)  # 获取原子的三维坐标
        atom_coords[idx] = (pos.x, pos.y, pos.z)
    return atom_coords


def Mol2Graph(mol, max_nb=6):
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    atom_feature_matrix = np.zeros((n_atoms, atom_fdim), dtype=np.float32)
    bond_feature_matrix = np.zeros((n_bonds, bond_fdim), dtype=np.float32)
    atom_nb = np.zeros((n_atoms, max_nb), dtype=int)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=int)
    num_nbs = np.zeros((n_atoms,), dtype=int)
    adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)
    atom_coords = extract_3d_coordinates(mol)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_feature_matrix[idx] = atom_features(atom)

    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bond_feat = bond_features(bond)
        bond_feature_matrix[idx] = bond_feat

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

    return atom_feature_matrix, bond_feature_matrix, atom_nb, bond_nb, num_nbs, adj_matrix, atom_coords


def save_graph_to_pt(graph_data, output_path):
    atom_feature_matrix, bond_feature_matrix, atom_nb, bond_nb, num_nbs, adj_matrix, atom_coords = graph_data

    node_features = torch.tensor(atom_feature_matrix, dtype=torch.float32)  # 原子特征
    edge_index = []
    edge_attr = []

    for i in range(len(bond_nb)):
        for j in range(num_nbs[i]):
            if bond_nb[i, j] != 0:
                edge_index.append([i, atom_nb[i, j]])
                edge_attr.append(bond_feature_matrix[bond_nb[i, j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    torch.save(data, output_path)


def process_and_save_graphs(input_folder, output_folder, max_nb=6):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for sdf_file in os.listdir(input_folder):
        if sdf_file.endswith('.sdf'):
            sdf_file_path = os.path.join(input_folder, sdf_file)
            molecule_name = os.path.splitext(sdf_file)[0]
            output_file_path = os.path.join(output_folder, f'{molecule_name}.pt')

            try:
                suppl = Chem.SDMolSupplier(sdf_file_path, sanitize=False)
                for mol in suppl:
                    if mol is None:
                        print(f"Invalid molecule in {sdf_file}. Skipping...")
                        continue

                    try:
                        Chem.SanitizeMol(mol)
                        Chem.Kekulize(mol, clearAromaticFlags=True)

                        graph_data = Mol2Graph(mol, max_nb=max_nb)

                        save_graph_to_pt(graph_data, output_file_path)
                        print(f"Saved graph for {molecule_name} to {output_file_path}")
                    except Exception as e:
                        print(f"Error processing molecule {molecule_name}: {e}. Skipping...")
                        continue
            except Exception as e:
                print(f"Error reading {sdf_file_path}: {e}. Skipping...")


input_folder = './Data/pdb_files/ligand'
output_folder = './Data/site/data/ligand_graph'

process_and_save_graphs(input_folder, output_folder, max_nb=6)
