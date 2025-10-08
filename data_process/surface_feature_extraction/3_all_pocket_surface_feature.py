import os
import pickle
import numpy as np
import trimesh
import torch
from sklearn.neighbors import KDTree
from Bio.PDB import PDBParser


class FeatureProcessor():
    def __init__(self, pdbid, dir_opts={}):
        self.dir_opts = dir_opts
        self.pdbid = pdbid
        self.mesh, self.normalv, self.vert_info = self.load_surface_features()
        print(f"Loading alpha carbons for: {self.pdbid}")
        self.alpha_carbons, self.all_atoms, self.residue_to_index = self.load_alpha_carbons()

        print(f"Mapping residues to vertices for: {self.pdbid}")
        self.residue_to_vertex_mapping = self.map_residues_to_vertices()
        print(f"Getting curvature graph for: {self.pdbid}")
        self.curvature_graph = self.get_curvature_graph()
        print(f"Getting BOARD graph for: {self.pdbid}")
        self.BOARD_graph = self.get_BOARD()

    def load_surface_features(self):  # 通过ply文件加载蛋白质的表面信息
        ply_file = os.path.join(self.dir_opts['ply_files'], f"{self.pdbid}.ply")

        if not os.path.exists(ply_file):
            raise FileNotFoundError(f"PLY file not found: {ply_file}")

        try:
            mesh = trimesh.load(ply_file)
        except Exception as e:
            raise ValueError(f"Error loading PLY file: {ply_file}, {e}")

        xyzrn_file = os.path.join(self.dir_opts['xyzrn_files'], f"{self.pdbid}.xyzrn")

        if not os.path.exists(xyzrn_file):
            raise FileNotFoundError(f"XYZRN file not found: {xyzrn_file}")

        vert_info = []
        with open(xyzrn_file, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"Skipping malformed line in XYZRN file: {line.strip()}")
                        continue
                    x, y, z, r, element = parts[:5]
                    vert_info.append([float(x), float(y), float(z), float(r), element])
                except ValueError as ve:
                    print(f"Error parsing line in XYZRN file: {line.strip()}")
                    raise ve

        normalv = mesh.vertex_normals
        return mesh, normalv, vert_info

    def load_alpha_carbons(self):
        pdb_file = os.path.join(self.dir_opts['pdb_files'], f"{self.pdbid}.pdb")

        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(self.pdbid, pdb_file)
        alpha_carbons = []
        all_atoms = []
        residue_to_index = {}  # 残基编号到索引的映射
        index = 0  # 用于构建映射表的索引

        # 获取所有链标识符
        protein_chains = [chain.id for model in structure for chain in model]

        for model in structure:
            for chain in model:
                if chain.id in protein_chains:  # 自动识别所有链
                    for residue in chain:
                        resid = residue.id[1]
                        # 为每个残基创建索引映射
                        residue_to_index[f"{chain.id}_{resid}"] = index
                        index += 1
                        for atom in residue:
                            all_atoms.append((atom.get_coord(), chain.id, residue.id[1], atom.name))
                            if atom.name == 'CA':
                                alpha_carbons.append((atom.get_coord(), chain.id, residue.id[1]))

        return alpha_carbons, all_atoms, residue_to_index

    def map_residues_to_vertices(self):
        residue_to_vertex_mapping = {}
        alpha_coords = np.array([coord[0] for coord in self.alpha_carbons])
        kdtree = KDTree(alpha_coords)

        for i, vert in enumerate(self.mesh.vertices):
            dist, idx = kdtree.query([vert], k=1)
            nearest_res = self.alpha_carbons[idx[0][0]]
            chain_res_id = f"{nearest_res[1]}_{nearest_res[2]}"

            # 设置一个合理的距离阈值（例如8.0Å）
            if dist[0][0] < 8.0:
                if chain_res_id not in residue_to_vertex_mapping:
                    residue_to_vertex_mapping[chain_res_id] = []
                residue_to_vertex_mapping[chain_res_id].append(i)

        self.residue_to_vertex_mapping = residue_to_vertex_mapping
        return residue_to_vertex_mapping

    def check_mesh_repair(self):  # 使用trimesh修复网格
        mesh = trimesh.Trimesh(vertices=self.mesh.vertices, faces=self.mesh.faces)
        # 使用 trimesh 的修复功能
        mesh.process(validate=True)
        vertices = mesh.vertices
        faces = mesh.faces
        return vertices, faces

    def get_curvature_graph(self):  # 获取曲率图，计算网格顶点的主曲率和高斯曲率，并返回一个包含这些曲率信息的张量
        vertices, faces = self.check_mesh_repair()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 计算曲率
        mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius=1.0)
        gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=1.0)

        curvature_graph = np.vstack([mean_curvature, gaussian_curvature]).T
        return torch.tensor(curvature_graph, dtype=torch.float32)

    def get_BOARD(self):  # 获取 BOARD 图，计算 BOARD 特征，并返回一个包含这些信息的张量
        tree = KDTree(self.mesh.vertices)
        indices = tree.query_radius(self.mesh.vertices, r=12)
        BOARD = []
        for index in range(self.mesh.vertices.shape[0]):
            points = indices[index]
            if len(points) == 0:
                BOARD.append(0)
                continue
            board_coord = self.mesh.vertices[points]
            signed_dis = np.mean((board_coord - self.mesh.vertices[index]) * self.normalv[index])
            BOARD.append(signed_dis)
        BOARD = np.array(BOARD)
        BOARD = (BOARD - np.min(BOARD)) / (np.max(BOARD) - np.min(BOARD))
        return torch.tensor(BOARD, dtype=torch.float32)[:, None]

    def get_surface_feature_for_residue(self, res_id):  # 为每个残基生成表面特征，并返回其平均值
        if res_id not in self.residue_to_vertex_mapping:
            return torch.zeros(8)  # 如果没有对应的顶点，返回全零张量

        vertex_indices = self.residue_to_vertex_mapping[res_id]

        # 提取这些顶点的曲率信息
        curvature_features = self.curvature_graph[vertex_indices]
        mean_curvature = curvature_features[:, 0]
        gaussian_curvature = curvature_features[:, 1]


        # 计算表面特征的平均值或其他统计值（例如，平均曲率，法向量的平均值）
        avg_mean_curvature = torch.mean(mean_curvature)
        avg_gaussian_curvature = torch.mean(gaussian_curvature)

        # 其他可能的表面特征（如Shape Index, Curvedness等）
        shape_index = (2 / torch.pi) * torch.atan2(avg_mean_curvature, avg_gaussian_curvature)
        curvedness = torch.sqrt(avg_mean_curvature ** 2 + avg_gaussian_curvature ** 2)

        # 组合成8维特征向量
        surface_feature = torch.cat([torch.tensor([avg_mean_curvature, avg_gaussian_curvature]),
                                     torch.tensor([shape_index, curvedness])], dim=0)
        return surface_feature

    def get_chemical_feature_for_residue(self, res_id):  # 为每个残基生成化学特征的独热编码，并返回其平均值
        atom_onehot = {"C": [1, 0, 0, 0, 0, 0], "H": [0, 1, 0, 0, 0, 0], "O": [0, 0, 1, 0, 0, 0],
                       "N": [0, 0, 0, 1, 0, 0], "S": [0, 0, 0, 0, 1, 0], "Other": [0, 0, 0, 0, 0, 1]}
        atoms = [atom for atom in self.all_atoms if f"{atom[1]}_{atom[2]}" == res_id]
        if not atoms:
            return torch.zeros(6)
        atom_types = torch.tensor([atom_onehot.get(atom[3][0], atom_onehot["Other"]) for atom in atoms],
                                  dtype=torch.float32)
        return torch.mean(atom_types, axis=0)

    def get_data(self):
        # 初始化蛋白质的残基特征张量，shape 为 (len(alpha_carbons), 14)
        surface_features = torch.zeros((len(self.alpha_carbons), 14))

        for chain_res_id, vertices in self.residue_to_vertex_mapping.items():
            # 检查残基是否有映射索引
            if chain_res_id not in self.residue_to_index:
                print(f"Skipping residue {chain_res_id} as it is not in residue_to_index")
                continue

            seq_index = self.residue_to_index[chain_res_id]  # 使用字典映射来获取索引
            # print(seq_index)

            surface_feature = self.get_surface_feature_for_residue(chain_res_id)
            assert surface_feature.shape[
                       0] == 4, f"Surface feature size mismatch: expected 8, got {surface_feature.shape[0]}"

            # 获取化学特征 (6维)
            chemical_feature = self.get_chemical_feature_for_residue(chain_res_id)
            assert chemical_feature.shape[
                       0] == 6, f"Chemical feature size mismatch: expected 6, got {chemical_feature.shape[0]}"

            surface_features[seq_index] = torch.cat((chemical_feature, surface_feature), dim=0)


        return surface_features

    def extract_pocket_features(self, pocket_residues, output_dir):
        pocket_features = []

        full_features = self.get_data()

        for res_id in pocket_residues:
            # 格式化 pocket_residues 为 A_30 格式
            formatted_res_id = res_id[0] + "_" + res_id[1:]  # 例如将 "A30" 转为 "A_30"

            if formatted_res_id in self.residue_to_index:
                seq_index = self.residue_to_index[formatted_res_id]  # 获取该残基的序列索引
                pocket_features.append(full_features[seq_index])  # 提取特征

        pocket_features = torch.stack(pocket_features)
        return pocket_features


    def save_pocket_features(self, pocket_features, output_file):
        # 保存口袋部分的特征
        with open(output_file, 'wb') as f:
            pickle.dump(pocket_features, f)
        print(f"Pocket features saved to {output_file}")

    def save_residue_features(self, all_features, output_dir):
        output_file = os.path.join(output_dir, "all_pocket_features.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(all_features, f)
        print(f"Saved all residue features to: {output_file}")


def load_proteins(txt_file):
    with open(txt_file, 'r') as f:
        proteins = f.read().splitlines()
    return proteins

if __name__ == '__main__':
    dir_opts = {
        'protein_list': './Data/dti/pocket_pdbid.txt',  # 存储所有蛋白质 pdbid 的 txt 文件
        'pocket_residue_file': '.Data/affinity/pocket_residue_chain_info.txt',  # 存储口袋残基信息的 txt 文件
        'data_output': './data_process/pocket/surface_out',
        'pdb_files': './Data/protein_pdb',
        'ply_files': './data_process/surface/protein/ply',
        'xyzrn_files': './data_process/surface/protein/xyzrn'
    }

    proteins = load_proteins(dir_opts['protein_list'])  # 加载所有 pdbid
    pocket_residues_info = {}  # 用于存储每个蛋白质的口袋残基信息
    all_features = {}
    all_pocket_features = {}  # 用于存储所有蛋白质口袋特征

    # 加载口袋残基信息
    pocket_residues_mapping = {}
    with open(dir_opts['pocket_residue_file'], 'r') as f:
        for line in f:
            pdbid, residues = line.strip().split(": ")
            residues_list = residues.split(",")
            pocket_residues_mapping[pdbid] = residues_list  # 生成 pdbid -> 口袋残基映射

    # 遍历所有的蛋白质 pdbid
    for pdbid in proteins:
        try:
            processor = FeatureProcessor(pdbid, dir_opts=dir_opts)

            # 获取所有表面特征（包括化学特征等）
            features = processor.get_data()
            all_features[pdbid] = features.numpy()

            # 获取对应 pdbid 的口袋残基特征
            if pdbid in pocket_residues_mapping:
                pocket_residues = pocket_residues_mapping[pdbid]  # 获取该 pdbid 对应的口袋残基
                pocket_features = processor.extract_pocket_features(pocket_residues, dir_opts['data_output'])

                # 将口袋特征保存到字典中，使用 pdbid 作为键
                all_pocket_features[pdbid] = pocket_features

                # 将口袋残基信息添加到字典中
                pocket_residues_info[pdbid] = pocket_residues

        except Exception as e:
            print(f"Error processing {pdbid}: {e}")

    # 保存所有残基的特征到 pkl 文件
    processor.save_residue_features(all_features, dir_opts['data_output'])

    # 保存口袋特征到一个统一的 pkl 文件
    pocket_features_file = os.path.join(dir_opts['data_output'], 'all_pocket_features.pkl')
    with open(pocket_features_file, 'wb') as f:
        pickle.dump(all_pocket_features, f)

    # 保存口袋残基信息到 txt 文件
    pocket_residue_info_file = os.path.join(dir_opts['data_output'], "pocket_residue_info.txt")
    with open(pocket_residue_info_file, 'w') as f:
        for pdbid, residues in pocket_residues_info.items():
            residues_str = ",".join(residues)
            f.write(f"{pdbid}: {residues_str}\n")

    print(f"Saved pocket features to: {pocket_features_file}")
    print(f"Saved pocket residue information to: {pocket_residue_info_file}")
