from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
# import dgl
from scipy.spatial import distance_matrix  # 导入了SciPy库和NumPy库，用于科学计算和处理数组
import numpy as np
import pandas as pd
import os
import pickle
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm
import Bio.PDB
from Bio.SeqUtils import seq1
import h5py
from functools import partial
from scipy.spatial.transform import Rotation as R
import warnings
import multiprocessing

warnings.filterwarnings('ignore')  # 禁止了警告输出，即在运行时不会显示警告信息


# 定义药物特征维度
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
    num_nbs_mat = np.zeros((n_atoms, max_nb), dtype=object)
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

    for i in range(len(num_nbs)):
        num_nbs_mat[i, :num_nbs[i]] = 1

    return atom_feature_matrix, bond_feature_matrix, atom_nb, bond_nb, num_nbs, num_nbs_mat, atom_coords


def process_sdf_file(sdf_file_path, max_nb=6):

    suppl = Chem.SDMolSupplier(sdf_file_path, sanitize=False)  # 禁用自动 sanitize
    for mol in suppl:
        if mol is None:
            print(f"Invalid molecule in {sdf_file_path}. Skipping...")
            continue

        try:
            # 手动 sanitize，确保分子合法
            Chem.SanitizeMol(mol)

            # 尝试 kekulize 分子
            Chem.Kekulize(mol, clearAromaticFlags=True)

            # 转换为图形
            result = Mol2Graph(mol, max_nb=max_nb)
            yield result

        except Exception as e:
            print(f"Error processing molecule: {e}. Skipping...")
            continue


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


def build_protein_graph(pdbid, residue_df, pretrained_features, surface_features, k_neighbors=8, cutoff=12.0, sigma=1.0):
    protein_resnames = residue_df['res_name'].values  # 残基名称
    protein_coords = np.array(residue_df['c_alpha'].tolist())  # 残基坐标 (N x 3)

    if pdbid not in pretrained_features:
        raise KeyError(f"Pretrained features for {pdbid} not found in pretrained_features.")

    pretrained_feat = pretrained_features[pdbid]
    if pretrained_feat.shape[0] != len(protein_resnames):
        print(
            f"Skipping {pdbid} due to length mismatch: PDB sequence length ({len(protein_resnames)}) vs pretrained features length ({pretrained_feat.shape[0]}).")
        return None


    # 获取表面特征
    surf_feat = surface_features.get(pdbid, np.zeros((len(protein_coords), 14)))

    # 计算距离矩阵
    dist_matrix = np.linalg.norm(protein_coords[:, None, :] - protein_coords[None, :, :], axis=-1)
    # adj_matrix = (dist_matrix < cutoff).astype(np.float32)

    # 提取边对 (i, j)，满足距离条件
    edges = np.where(dist_matrix < cutoff)
    edges = list(zip(edges[0], edges[1]))

    edge_features = []
    atom_nb = [[] for _ in range(len(protein_coords))]  # 每个残基的邻居索引
    bond_nb = [[] for _ in range(len(protein_coords))]  # 每个残基的连边索引
    num_nbs = np.zeros(len(protein_coords), dtype=int)  # 每个残基的邻居数量
    neighbor_projections = []  # 每个残基的邻居在局部坐标系下的投影坐标

    for i, j in edges:
        if i == j:  # 排除自身到自身的边
            continue

        c_alpha_i = protein_coords[i]
        c_i = residue_df.iloc[i]['c']
        prev_c_alpha_i = protein_coords[i - 1] if i > 0 else None

        local_frame_i = compute_local_frame(c_alpha_i, c_i, prev_c_alpha_i)
        local_frame_j = compute_local_frame(protein_coords[j], residue_df.iloc[j]['c'],
                                            protein_coords[j - 1] if j > 0 else None)

        R_ij = compute_rotation_matrix(local_frame_i, local_frame_j)
        quaternion = rotation_matrix_to_quaternion(R_ij)

        edge_vector = protein_coords[j] - c_alpha_i
        x_proj = np.dot(edge_vector, local_frame_i[0])
        y_proj = np.dot(edge_vector, local_frame_i[1])
        z_proj = np.dot(edge_vector, local_frame_i[2])
        distance = np.linalg.norm(edge_vector)
        rbf_distance = np.exp(-distance ** 2 / (2 * sigma ** 2))

        edge_feature = np.array([
            x_proj, y_proj, z_proj, rbf_distance,
            quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        ])
        edge_features.append(edge_feature)

        atom_nb[i].append(j)
        bond_nb[i].append(len(edge_features) - 1)
        num_nbs[i] += 1

        neighbor_projection = np.dot(protein_coords[j] - c_alpha_i, np.column_stack(local_frame_i))
        neighbor_projections.append(neighbor_projection.tolist())

    # 将邻居信息转换为矩阵形式
    max_neighbors = max(num_nbs)
    atom_nb_matrix = np.zeros((len(protein_coords), max_neighbors), dtype=int)
    bond_nb_matrix = np.zeros((len(protein_coords), max_neighbors), dtype=int)

    for i in range(len(protein_coords)):
        atom_nb_matrix[i, :num_nbs[i]] = atom_nb[i]
        bond_nb_matrix[i, :num_nbs[i]] = bond_nb[i]

    num_nbs_mat = np.zeros((len(protein_coords), max_neighbors), dtype=int)
    for i in range(len(num_nbs)):
        num_nbs_mat[i, :num_nbs[i]] = 1

    # 返回蛋白质图结构
    return {
        'resnames': protein_resnames,
        'coords': protein_coords,
        'pretrained_features': pretrained_feat,
        'surface_features': surf_feat,
        'edge_features': edge_features,
        'atom_nb': atom_nb_matrix,
        'bond_nb': bond_nb_matrix,
        'num_nbs': num_nbs,
        'num_nbs_mat': num_nbs_mat,
    }



def compute_local_frame(c_alpha, c, prev_c_alpha=None):

    z_axis = c - c_alpha
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0., 0., 1.])
    else:
        z_axis = z_axis / z_norm

    # 处理xz平面基准向量
    if prev_c_alpha is not None:
        xz_plane = prev_c_alpha - c_alpha
        xz_norm = np.linalg.norm(xz_plane)
        if xz_norm < 1e-6:
            xz_plane = np.array([1., 0., 0.])  # 默认X轴方向
        else:
            xz_plane = xz_plane / xz_norm
    else:
        xz_plane = np.array([1., 0., 0.])  # 默认X轴方向

    # 确保正交性
    y_axis = np.cross(z_axis, xz_plane)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:  # 处理共线情况
        y_axis = np.array([0., 1., 0.])  # 默认Y轴方向
    else:
        y_axis = y_axis / y_norm

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    return x_axis, y_axis, z_axis


def compute_rotation_matrix(local_frame_i, local_frame_j):

    M_i = np.column_stack(local_frame_i)
    M_j = np.column_stack(local_frame_j)

    # 正交性检查
    if not np.allclose(M_i.T @ M_i, np.eye(3), atol=1e-3):
        M_i = np.eye(3)
    if not np.allclose(M_j.T @ M_j, np.eye(3), atol=1e-3):
        M_j = np.eye(3)

    R_ij = M_j @ M_i.T
    return R_ij


def rotation_matrix_to_quaternion(R_ij):

    det = np.linalg.det(R_ij)
    if not (0.99 < abs(det) < 1.01):
        R_ij = np.eye(3)

    try:
        rotation = R.from_matrix(R_ij)
        quaternion = rotation.as_quat()  # [x, y, z, w]
    except ValueError:
        quaternion = np.array([0., 0., 0., 1.])  # 单位四元数

    # 确保四元数归一化
    quaternion = quaternion / np.linalg.norm(quaternion)
    return quaternion


def build_hetero_graph(protein_coords, drug_coords, k_neighbors=6, sigma=1.0):

    protein_coords = np.array(protein_coords)
    drug_coords = np.array(drug_coords)

    distances = np.linalg.norm(protein_coords[:, None, :] - drug_coords[None, :, :], axis=-1)

    edge_features_residue_to_atom = []
    edge_features_atom_to_residue = []

    protein_neighbors = [[] for _ in protein_coords]
    drug_neighbors = [[] for _ in drug_coords]

    drug_to_protein_nearest = []
    for l_idx in range(len(drug_coords)):
        dists = np.linalg.norm(protein_coords - drug_coords[l_idx], axis=1)
        drug_to_protein_nearest.append(np.argmin(dists))

    # ===================== 蛋白质到药物的边 =====================
    for p_idx in range(len(protein_coords)):
        sorted_ligand_idxs = np.argsort(distances[p_idx])[:k_neighbors]

        if len(sorted_ligand_idxs) > 0:
            nearest_ligand = drug_coords[sorted_ligand_idxs[0]]
        else:
            nearest_ligand = protein_coords[p_idx] + [0, 0, 1]

        prev_ca = protein_coords[p_idx - 1] if p_idx > 0 else None

        local_frame_i = compute_local_frame(
            c_alpha=protein_coords[p_idx],
            c=nearest_ligand,
            prev_c_alpha=prev_ca
        )

        for l_idx in sorted_ligand_idxs:
            # ================= 药物原子局部坐标系 =================
            nearest_p_idx = drug_to_protein_nearest[l_idx]
            protein_ref = protein_coords[nearest_p_idx]

            drug_dists = np.linalg.norm(drug_coords - drug_coords[l_idx], axis=1)
            drug_dists[l_idx] = np.inf  # 排除自身
            nearest_drug_idx = np.argmin(drug_dists)
            drug_ref = drug_coords[nearest_drug_idx] if len(drug_coords) > 1 else drug_coords[l_idx] + [1, 0, 0]

            # 防止基准点重合
            if np.allclose(drug_coords[l_idx], drug_ref):
                drug_ref += 0.1 * np.random.randn(3)

            local_frame_j = compute_local_frame(
                c_alpha=drug_coords[l_idx],
                c=protein_ref,  # Z轴指向最近的蛋白质
                prev_c_alpha=drug_ref  # XZ平面基准
            )

            try:
                R_ij = compute_rotation_matrix(local_frame_i, local_frame_j)
                quaternion = rotation_matrix_to_quaternion(R_ij)
            except:
                quaternion = np.array([0., 0., 0., 1.])

                # ================= 边向量计算 =================
            edge_vector = drug_coords[l_idx] - protein_coords[p_idx]
            edge_distance = np.linalg.norm(edge_vector)

            edge_vector = drug_coords[l_idx] - protein_coords[p_idx]
            edge_distance = np.linalg.norm(edge_vector)

            # 投影到蛋白质坐标系
            edge_local_i = np.array([
                np.dot(edge_vector, local_frame_i[0]),
                np.dot(edge_vector, local_frame_i[1]),
                np.dot(edge_vector, local_frame_i[2])
            ])

            # 投影到药物原子坐标系（反向边）
            edge_local_j = np.array([
                np.dot(-edge_vector, local_frame_j[0]),
                np.dot(-edge_vector, local_frame_j[1]),
                np.dot(-edge_vector, local_frame_j[2])
            ])

            # ================= 构建边特征 =================
            residue_to_atom_feature = np.concatenate([
                edge_local_i,
                [np.exp(-edge_distance ** 2 / (2 * sigma ** 2))],
                quaternion
            ])

            atom_to_residue_feature = np.concatenate([
                edge_local_j,
                [np.exp(-edge_distance ** 2 / (2 * sigma ** 2))],
                quaternion
            ])
            edge_idx = len(edge_features_residue_to_atom)
            edge_features_residue_to_atom.append(residue_to_atom_feature)
            edge_features_atom_to_residue.append(atom_to_residue_feature)

            protein_neighbors[p_idx].append(l_idx)
            drug_neighbors[l_idx].append(p_idx)

    # ===================== 邻居矩阵填充 =====================
    # 蛋白质邻居矩阵 (n_protein, k)
    protein_neighbors_matrix = np.full((len(protein_coords), k_neighbors), 0, dtype=int)
    for i, neighbors in enumerate(protein_neighbors):
        n_neighbors = min(len(neighbors), k_neighbors)
        if n_neighbors > 0:
            protein_neighbors_matrix[i, :n_neighbors] = neighbors[:n_neighbors]

    # 药物邻居矩阵 (m_drug, k)
    drug_neighbors_matrix = np.full((len(drug_coords), k_neighbors), 0, dtype=int)
    for i, neighbors in enumerate(drug_neighbors):
        # 按距离排序并截断
        if len(neighbors) > k_neighbors:
            dists = distances[neighbors, i]
            sorted_idx = np.argsort(dists)[:k_neighbors]
            neighbors = [neighbors[idx] for idx in sorted_idx]

        n_neighbors = min(len(neighbors), k_neighbors)
        if n_neighbors > 0:
            drug_neighbors_matrix[i, :n_neighbors] = neighbors[:n_neighbors]

    # ===================== 边索引矩阵 =====================
    # 蛋白质边索引 (n_protein, k)
    protein_bonds_matrix = np.full((len(protein_coords), k_neighbors), 0, dtype=int)
    edge_counter = 0
    for i, neighbors in enumerate(protein_neighbors):
        for j in range(min(len(neighbors), k_neighbors)):
            protein_bonds_matrix[i, j] = edge_counter
            edge_counter += 1

    # 药物边索引 (m_drug, k)
    drug_bonds_matrix = np.full((len(drug_coords), k_neighbors), 0, dtype=int)
    edge_counter = 0
    for i, neighbors in enumerate(drug_neighbors):
        for j in range(min(len(neighbors), k_neighbors)):
            drug_bonds_matrix[i, j] = edge_counter
            edge_counter += 1


    protein_neighbor_counts = np.array([len(n) for n in protein_neighbors], dtype=np.int32)
    drug_neighbor_counts = np.array([len(n) for n in drug_neighbors], dtype=np.int32)

    protein_num_nbs_mat = np.zeros((len(protein_coords), k_neighbors), dtype=int)
    drug_num_nbs_mat = np.zeros((len(drug_coords), k_neighbors), dtype=int)

    for i in range(len(protein_neighbor_counts)):
        protein_num_nbs_mat[i, :protein_neighbor_counts[i]] = 1

    for i in range(len(drug_neighbor_counts)):
        drug_num_nbs_mat[i, :drug_neighbor_counts[i]] = 1

    # ===================== 最终数据结构 =====================
    hetero_graph_data = {
        'edge_features_residue_to_atom': np.array(edge_features_residue_to_atom, dtype=np.float32),
        'edge_features_atom_to_residue': np.array(edge_features_atom_to_residue, dtype=np.float32),
        'protein_neighbors': protein_neighbors_matrix,
        'drug_neighbors': drug_neighbors_matrix,
        'protein_bonds': protein_bonds_matrix,
        'drug_bonds': drug_bonds_matrix,
        'protein_neighbor_counts': protein_neighbor_counts,
        'drug_neighbor_counts': drug_neighbor_counts,
        'protein_num_nbs_mat': protein_num_nbs_mat,
        'drug_num_nbs_mat': drug_num_nbs_mat,

    }

    # 最终数值检查
    for key in ['edge_features_residue_to_atom', 'edge_features_atom_to_residue']:
        if np.any(np.isnan(hetero_graph_data[key])):
            invalid_count = np.isnan(hetero_graph_data[key]).sum()
            print(f"Warning: {invalid_count} NaN values found in {key}, replacing with zeros")
            hetero_graph_data[key] = np.nan_to_num(hetero_graph_data[key])

    return hetero_graph_data


def load_protein_pretrained_features(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        protein_features = {}
        for key in f.keys():
            protein_features[key] = f[key][:]
    return protein_features


def load_protein_surface_features():
    with open (surface_feature_file, 'rb') as f:
        protein_surf_features = pickle.load(f)
    return protein_surf_features



def get_pdbid_to_affinity(index_file):
    pdbid_to_measure, pdbid_to_value = {}, {}  # value: -log [M]
    with open(index_file) as f:
        count_error = 0
        for line in f.readlines():
            if line[0] != '#':
                lines = line.split('/')[0].strip().split('  ')
                pdbid = lines[0]
                if '<' in lines[3] or '>' in lines[3] or '~' in lines[3]:
                    count_error += 1
                    continue
                measure = lines[3].split('=')[0]
                value = float(lines[3].split('=')[1][:-2])
                unit = lines[3].split('=')[1][-2:]
                if unit == 'nM':
                    pvalue = -np.log10(value) + 9
                elif unit == 'uM':
                    pvalue = -np.log10(value) + 6
                elif unit == 'mM':
                    pvalue = -np.log10(value) + 3
                elif unit == 'pM':
                    pvalue = -np.log10(value) + 12
                elif unit == 'fM':
                    pvalue = -np.log10(value) + 15
                else:
                    print(unit)
                pdbid_to_measure[pdbid] = measure
                pdbid_to_value[pdbid] = pvalue
    print('count_error not = measurement', count_error)
    return pdbid_to_measure, pdbid_to_value


def main(pdbid_list_file, pdb_folder, sdf_folder, output_folder, h5_file_path, surface_feature_file, index_file):
    with open(pdbid_list_file, 'r') as f:
        pdbids = [line.strip() for line in f.readlines()]


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    protein_pretrained_features = load_protein_pretrained_features(h5_file_path)
    protein_surface_features = load_protein_surface_features()
    pdbid_to_measure, pdbid_to_value = get_pdbid_to_affinity(index_file)


    # 初始化一个列表，用来保存每个字段的所有数据
    coords = []
    pretrained_features = []
    surface_features = []
    edge_features = []
    # adj_matrix = []
    atom_nb = []
    bond_nb = []
    num_nbs = []
    p_num_nbs_mar = []
    a_fea = []
    b_fea = []
    anb = []
    bnb = []
    nbs = []
    num_nbs_mar =[]
    # adj_mar = []
    a_coords = []
    affinity_labels = []

    # 异质图特征
    edge_features_residue_to_atom = []
    edge_features_atom_to_residue = []
    p_nei =[]
    d_nei =[]
    p_bond_nei = []
    d_bond_nei = []
    p_nei_nbs = []
    d_nei_nbs = []
    p_nbs_mar = []
    d_nbs_mar = []
    # hetero_adj = []

    # processed_pdbids = []

    for pdbid in pdbids:
        # 构建PDB文件和SDF文件的路径
        pdb_file_path = os.path.join(pdb_folder, f"{pdbid}.pdb")
        if not isinstance(pdb_file_path, str):
            raise TypeError(f"Expected pdb_file_path to be a string, got {type(pdb_file_path)}")
        if not os.path.exists(pdb_file_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")

        sdf_file_path = os.path.join(sdf_folder, f"{pdbid}_ligand.sdf")

        # 检查文件是否存在
        if not os.path.exists(pdb_file_path) or not os.path.exists(sdf_file_path):
            print(f"Files for PDB ID {pdbid} not found. Skipping...")
            continue

        # 获取蛋白质的预训练特征和表面特征
        protein_pretrained_feat = protein_pretrained_features.get(pdbid, None)
        protein_surf_feat = protein_surface_features.get(pdbid, None)

        # 检查特征长度是否一致
        if protein_pretrained_feat is None or protein_surf_feat is None:
            print(f"Missing features for PDB ID {pdbid}. Skipping...")
            continue

        if len(protein_pretrained_feat) != len(protein_surf_feat):
            print(f"Feature length mismatch for PDB ID {pdbid}. Skipping...")
            continue

        # 获取亲和力标签
        label = pdbid_to_value.get(pdbid, None)

        # 检查亲和力标签是否为 NaN 或 None
        if label is None or np.isnan(label):  # 判断亲和力标签为 NaN 或 None
            print(f"Affinity label is None or NaN for PDB ID {pdbid}. Skipping this PDB ID...")
            continue  # 跳过这个 pdbid，不处理数据

        # 构建药物分子图
        drug_graph_list = list(process_sdf_file(sdf_file_path))
        if len(drug_graph_list) > 0:
            drug_graph = drug_graph_list[0]
            if isinstance(drug_graph[6], dict):
                drug_coords = np.array(list(drug_graph[6].values()))  # 转换字典为 NumPy 数组
            else:
                drug_coords = np.array(drug_graph[6])
        else:
            print(f"No valid molecules found in {sdf_file_path}. Skipping...")
            continue

        residue_df, res_list = parse_pdb(pdb_file_path)  # 解包 parse_pdb 的返回值
        protein_graph = build_protein_graph(
            pdbid, residue_df, protein_pretrained_features, protein_surface_features
        )

        if protein_graph is None:
            print(f"Skipping {pdbid} as protein_graph is None.")
            continue  # 跳过当前 PDB 文件的后续处理


        # 将蛋白质和药物图的各个字段堆叠到对应的列表中
        coords.append(np.array(protein_graph['coords']))
        pretrained_features.append(np.array(protein_graph['pretrained_features']))
        surface_features.append(np.array(protein_graph['surface_features']))
        edge_features.append(np.array(protein_graph['edge_features']))
        # adj_matrix.append(np.array(protein_graph['adj_matrix']))
        atom_nb.append(np.array(protein_graph['atom_nb']))
        bond_nb.append(np.array(protein_graph['bond_nb']))
        num_nbs.append(np.array(protein_graph['num_nbs']))
        p_num_nbs_mar.append(np.array(protein_graph['num_nbs_mat']))

        # 药物图的字段
        a_fea.append(np.array(drug_graph[0]))
        b_fea.append(np.array(drug_graph[1]))
        anb.append(np.array(drug_graph[2]))
        bnb.append(np.array(drug_graph[3]))
        nbs.append(np.array(drug_graph[4]))
        num_nbs_mar.append(np.array(drug_graph[5]))
        # adj_mar.append(np.array(drug_graph[5]))
        a_coords.append(np.array(drug_coords))

        # 亲和力标签
        affinity_labels.append(label)

        # 构建异质图
        hetero_graph = build_hetero_graph(protein_graph['coords'], drug_coords)


        edge_features_residue_to_atom.append(hetero_graph['edge_features_residue_to_atom'])
        edge_features_atom_to_residue.append(hetero_graph['edge_features_atom_to_residue'])
        p_nei.append(hetero_graph['protein_neighbors'])
        d_nei.append(hetero_graph['drug_neighbors'])
        p_bond_nei.append(hetero_graph['protein_bonds'])
        d_bond_nei.append(hetero_graph['drug_bonds'])
        p_nei_nbs.append(hetero_graph['protein_neighbor_counts'])
        d_nei_nbs.append(hetero_graph['drug_neighbor_counts'])
        p_nbs_mar.append(hetero_graph['protein_num_nbs_mat'])
        d_nbs_mar.append(hetero_graph['drug_num_nbs_mat'])
        # hetero_adj.append(hetero_graph['adjacency_matrix'])


        # processed_pdbids.append(pdbid)

    # 合并所有字段的数据为一个大的numpy数组（按列拼接）
    data_pack = [
        np.array(pretrained_features, dtype=object),
        np.array(surface_features, dtype=object),
        np.array(edge_features, dtype=object),
        np.array(atom_nb, dtype=object),
        np.array(bond_nb, dtype=object),
        np.array(num_nbs, dtype=object),
        np.array(p_num_nbs_mar, dtype=object),

        np.array(a_fea, dtype=object),
        np.array(b_fea, dtype=object),
        np.array(anb, dtype=object),
        np.array(bnb, dtype=object),
        np.array(nbs, dtype=object),
        np.array(num_nbs_mar, dtype=object),
        np.array(affinity_labels),

        np.array(edge_features_residue_to_atom, dtype=object),
        np.array(edge_features_atom_to_residue, dtype=object),
        np.array(p_nei, dtype=object),
        np.array(d_nei, dtype=object),
        np.array(p_bond_nei, dtype=object),
        np.array(d_bond_nei, dtype=object),
        np.array(p_nei_nbs, dtype=object),
        np.array(d_nei_nbs, dtype=object),
        np.array(p_nbs_mar, dtype=object),
        np.array(d_nbs_mar, dtype=object),
    ]

    output_file_path = os.path.join(output_folder, "10_surf_train1_graph.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump(data_pack, f)

    print(f"All graphs and hetero graphs saved to {output_file_path}")


if __name__ == "__main__":

    pdbid_list_file = './Data/affinity/train.txt'  # PDB ID列表文件的路径
    pdb_folder = './Data/pdb_files/pocket_pdb'  # PDB文件夹的路径
    sdf_folder = './Data/pdb_files/ligand_sdf'  # SDF文件夹的路径
    output_folder = './Data/affinity'  # 输出文件夹的路径
    h5_file_path = './Data/ProT5/pocket_chains.h5'  # 预训练特征文件的路径
    surface_feature_file = './data_process/all_pocket_features.pkl'  # 表面特征文件的路径
    index_file = './Data/pdbbind_index/INDEX_general_PL.2020'  # 索引文件的路径

    # 调用主函数
    main(pdbid_list_file, pdb_folder, sdf_folder, output_folder, h5_file_path, surface_feature_file, index_file)

