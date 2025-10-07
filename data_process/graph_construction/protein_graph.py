import os
import numpy as np
import pandas as pd
import torch
import pickle
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import h5py
import warnings
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings('ignore')

def parse_pdb(file_path):
    res_dict = {
        'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
        'TRP': 'W', 'CYS': 'C', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H',
        'ASP': 'D', 'GLU': 'E', 'LYS': 'K', 'ARG': 'R'
    }

    pdb_res = pd.DataFrame(columns=['ID', 'atom', 'res', 'res_id', 'xyz', 'mass', 'is_sidechain'])
    res_list = []

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                atom_type = line[76:78].strip()
                res_name = line[17:20].strip()
                if res_name not in res_dict:
                    continue

                chain_id = line[21].strip()
                res_seq = line[22:27].strip()
                res_id = f"{chain_id}{res_seq}"
                atom_name = line[12:16].strip()
                x, y, z = map(float, (line[30:38], line[38:46], line[46:54]))
                is_sidechain = 0 if atom_name in ['N', 'CA', 'C', 'O', 'H'] else 1

                atom_info = {
                    'ID': len(pdb_res),
                    'atom': atom_name,
                    'res': res_dict[res_name],
                    'res_id': res_id,
                    'xyz': np.array([x, y, z]),
                    'is_sidechain': is_sidechain
                }
                pdb_res = pd.concat([pdb_res, pd.DataFrame([atom_info])], ignore_index=True)

    residue_info = []
    unique_res_ids = pdb_res['res_id'].unique()
    for res_id in unique_res_ids:
        residue_atoms = pdb_res[pdb_res['res_id'] == res_id]
        residue_name = residue_atoms['res'].iloc[0]

        # Handle residues without Cα atoms
        ca_atom = residue_atoms[residue_atoms['atom'] == 'CA']
        if len(ca_atom) == 0:
            xyz = residue_atoms['xyz'].values
            c_alpha = np.mean(xyz, axis=0)
        else:
            c_alpha = ca_atom['xyz'].values[0]

        c_atom = residue_atoms[residue_atoms['atom'] == 'C']
        if len(c_atom) == 0:
            continue  # Skip residues without C atoms
        c = c_atom['xyz'].values[0]

        residue_info.append({
            'res_id': res_id,
            'res_name': residue_name,
            'c_alpha': c_alpha,
            'c': c
        })
        res_list.append(res_id)

    residue_df = pd.DataFrame(residue_info)

    return residue_df, res_list


def compute_local_frame(c_alpha, c, prev_c_alpha=None):
    z_axis = c - c_alpha
    z_axis = z_axis / np.linalg.norm(z_axis)

    if prev_c_alpha is not None:
        xz_plane = prev_c_alpha - c_alpha
    else:
        xz_plane = np.array([1, 0, 0])
    xz_plane = xz_plane / np.linalg.norm(xz_plane)

    y_axis = np.cross(z_axis, xz_plane)
    y_axis = y_axis / np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    return x_axis, y_axis, z_axis


def compute_rotation_matrix(local_frame_i, local_frame_j):
    # local_frame_i 和 local_frame_j 是 (x_axis, y_axis, z_axis) 的元组
    x_axis_i, y_axis_i, z_axis_i = local_frame_i
    x_axis_j, y_axis_j, z_axis_j = local_frame_j

    # 构建方向矩阵
    M_i = np.column_stack((x_axis_i, y_axis_i, z_axis_i))
    M_j = np.column_stack((x_axis_j, y_axis_j, z_axis_j))

    # 计算旋转矩阵
    R_ij = M_j @ M_i.T
    return R_ij


def rotation_matrix_to_quaternion(R_ij):
    # 将旋转矩阵转换为四元数
    rotation = R.from_matrix(R_ij)
    quaternion = rotation.as_quat()  # [x, y, z, w]
    return quaternion


def build_protein_graph(pdbid, residue_df, res_list, pretrained_features, surface_features, labels, cutoff=12.0, sigma=1.0):
    protein_resnames = residue_df['res_name'].values
    protein_coords = np.array(residue_df['c_alpha'].tolist())

    if pdbid not in pretrained_features:
        raise KeyError(f"Pretrained features for {pdbid} not found in pretrained_features.")

    pretrained_feat = pretrained_features[pdbid]
    if pretrained_feat.shape[0] != len(protein_resnames):
        raise ValueError(
            f"Mismatch between PDB sequence length ({len(protein_resnames)}) and pretrained features length ({pretrained_feat.shape[0]})."
        )

    if pdbid not in surface_features:
        raise KeyError(f"Surface features for {pdbid} not found in surface_features.")

    surface_feat = surface_features[pdbid]
    if surface_feat.shape[0] != len(protein_resnames):
        raise ValueError(
            f"Mismatch between PDB sequence length ({len(protein_resnames)}) and surface features length ({surface_feat.shape[0]})."
        )

    # 计算距离矩阵
    dist_matrix = np.sqrt(((protein_coords[:, None, :] - protein_coords[None, :, :]) ** 2).sum(axis=2))
    edges = np.where(dist_matrix < cutoff)
    edges = list(zip(edges[0], edges[1]))

    # 过滤掉 i == j 的边
    edges = [(i, j) for i, j in edges if i != j]

    edge_features = []
    for i, j in edges:
        c_alpha_i = protein_coords[i]
        c_i = residue_df.iloc[i]['c']
        prev_c_alpha_i = protein_coords[i - 1] if i > 0 else None

        local_frame_i = compute_local_frame(c_alpha_i, c_i, prev_c_alpha_i)
        local_frame_j = compute_local_frame(protein_coords[j], residue_df.iloc[j]['c'], protein_coords[j - 1] if j > 0 else None)

        # 计算旋转矩阵
        R_ij = compute_rotation_matrix(local_frame_i, local_frame_j)
        quaternion = rotation_matrix_to_quaternion(R_ij)

        edge_vector = protein_coords[j] - c_alpha_i
        x_proj = np.dot(edge_vector, local_frame_i[0])
        y_proj = np.dot(edge_vector, local_frame_i[1])
        z_proj = np.dot(edge_vector, local_frame_i[2])
        distance = np.linalg.norm(edge_vector)
        rbf_distance = np.exp(-distance ** 2 / (2 * sigma ** 2))

        # 将四元数添加到边特征中
        edge_feature = np.array(
            [x_proj, y_proj, z_proj, rbf_distance, quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        edge_features.append(edge_feature)

    # 转换为 PyTorch 张量
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    x = torch.tensor(pretrained_feat, dtype=torch.float)
    surface_x = torch.tensor(surface_feat, dtype=torch.float)

    # 将标签转换为张量
    y = torch.tensor([int(label) for label in labels], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pdbid=pdbid, surface_x=surface_x)


def load_protein_pretrained_features(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        protein_features = {}
        for key in f.keys():
            protein_features[key] = f[key][:]
    return protein_features


def load_surface_features(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        surface_features = pickle.load(f)
    return surface_features


def load_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        lines = f.readlines()

    current_pdbid = None
    current_labels = []

    for line in lines:
        if line.startswith(">"):
            if current_pdbid:

                labels[current_pdbid] = ''.join(current_labels).strip()
            current_pdbid = line[1:].strip()
            current_labels = []
        else:
            if all(char in "01" for char in line.strip()):
                current_labels.append(line.strip())

    if current_pdbid:
        labels[current_pdbid] = ''.join(current_labels).strip()

    return labels


class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, root, pdbid_list, pdb_folder, h5_file_path, pkl_file_path, label_file, cutoff=12.0):
        self.pdbid_list = pdbid_list
        self.pdb_folder = pdb_folder
        self.h5_file_path = h5_file_path
        self.pkl_file_path = pkl_file_path
        self.label_file = label_file
        self.cutoff = cutoff
        self.pretrained_features = load_protein_pretrained_features(h5_file_path)
        self.surface_features = load_surface_features(pkl_file_path)
        self.labels = load_labels(label_file)
        super().__init__(root)
        self.data, self.slices = None, None

    @property
    def processed_file_names(self):
        return []

    def process(self):
        data_list = []
        skipped_pdbids = []

        for pdbid in tqdm(self.pdbid_list):
            pdb_file = os.path.join(self.pdb_folder, f"{pdbid}.pdb")

            if not os.path.exists(pdb_file):
                print(f"PDB file {pdbid} not found. Skipping.")
                skipped_pdbids.append(pdbid)
                continue

            residue_df, res_list = parse_pdb(pdb_file)

            if pdbid not in self.labels:
                print(f"Labels for {pdbid} not found. Skipping.")
                skipped_pdbids.append(pdbid)
                continue

            label = self.labels[pdbid]
            if len(label) != len(res_list):
                print(f"Length mismatch between labels and residues for {pdbid}. Skipping.")
                skipped_pdbids.append(pdbid)
                continue

            try:
                graph_data = build_protein_graph(pdbid, residue_df, res_list, self.pretrained_features, self.surface_features, label, cutoff=self.cutoff)
                # 保存每个蛋白质图到单独的.pt文件
                torch.save(graph_data, os.path.join(self.root, f"{pdbid}.pt"))
            except Exception as e:
                print(f"Error processing {pdbid}: {e}")
                skipped_pdbids.append(pdbid)

        # Save the order of processed PDB IDs
        with open(os.path.join(self.root, "1processed_pdbids.txt"), "w") as f:
            for pdbid in self.pdbid_list:
                if pdbid not in skipped_pdbids:
                    f.write(f"{pdbid}\n")

        # Save the skipped PDB IDs
        with open(os.path.join(self.root, "5skipped_pdbids.txt"), "w") as f:
            for pdbid in skipped_pdbids:
                f.write(f"{pdbid}\n")

if __name__ == "__main__":
    pdbid_list_file = "pdbid.txt"
    pdb_folder = "./Data/pdb_files/protein_pdb"
    output_folder = "./Data/site/protein_graph/"   # or ./Data/dti/protein_graph/
    h5_file_path = "./Data/pretrained/protein_pretrained_feature.h5"
    pkl_file_path = "./data_process/all_protein_surface_features.pkl"
    label_file = "./Data/site_labels.txt"

    with open(pdbid_list_file, 'r') as f:
        pdbid_list = f.read().splitlines()

    dataset = ProteinGraphDataset(root=output_folder, pdbid_list=pdbid_list, pdb_folder=pdb_folder, h5_file_path=h5_file_path, pkl_file_path=pkl_file_path, label_file=label_file)