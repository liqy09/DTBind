# -*- coding: utf-8 -*-
import os
import shutil
from subprocess import Popen, PIPE
import fcntl
from Bio.PDB import PDBParser, Selection
import pickle

# 原子半径
atom_radii = {
    "N": "1.540000",
    "O": "1.400000",
    "C": "1.740000",
    "H": "1.200000",
    "S": "1.800000",
    "P": "1.800000",
    "Z": "1.39",
    "X": "0.770000"
}

Amino_acid_type = [
    "ILE", "VAL", "LEU", "PHE", "CYS", "MET",
    "ALA", "GLY", "THR", "SER", "TRP", "TYR",
    "PRO", "HIS", "GLU", "GLN", "ASP", "ASN",
    "LYS", "ARG"
]

three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}

reduce_bin = os.path.expanduser("~/bin/reduce")


def protonate(pdb_file, dir_opts={}):
    """
    质子化 PDB 文件并保存输出文件。
    """
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]

    middle_file = os.path.join(dir_opts['protonated_pdb_dir'], pdb_id + '_protonated.pdb')
    out_pdb_file = os.path.join(dir_opts['protonated_pdb_dir'], pdb_id + '_protonated_final.pdb')

    if not os.path.exists(dir_opts['protonated_pdb_dir']):
        os.makedirs(dir_opts['protonated_pdb_dir'])

    # Remove protons first, in case the structure is already protonated
    args = [reduce_bin, "-Trim", pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    with open(middle_file, "w") as outfile:
        fcntl.flock(outfile, fcntl.LOCK_EX)
        outfile.write(stdout.decode('utf-8').rstrip())

    # Now add them again.
    args = [reduce_bin, "-HIS", middle_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    with open(out_pdb_file, "w") as outfile:
        fcntl.flock(outfile, fcntl.LOCK_EX)
        outfile.write(stdout.decode('utf-8').rstrip())

    return out_pdb_file


def extract_xyzrn(protonated_pdb_file, pair, dir_opts={}):
    """
    提取 XYZRN 文件
    """
    xyzrnfilename = os.path.join(dir_opts['xyzrn_dir'], pair + '.xyzrn')
    if not os.path.exists(dir_opts['xyzrn_dir']):
        os.makedirs(dir_opts['xyzrn_dir'])

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pair, protonated_pdb_file)

    out_list = []
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        chain = residue.get_parent().get_id()
        atomtype = name[0]
        coords = None
        if atomtype in atom_radii:
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            )
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, atomtype
            )
        if coords is not None:
            out_list.append((coords + " " + atom_radii[atomtype] + " 1 " + full_id + "\n"))
    with open(xyzrnfilename, "w") as outfile:
        outfile.writelines(out_list)


def get_seq(protonated_pdb_file, pair, dir_opts={}):
    """
    获取蛋白质序列

    Args:
        protonated_pdb_file (str): 质子化后的 PDB 文件路径。
        pair (str): 由蛋白质链信息组成的字符串，如 '1a0q_HL'。
        dir_opts (dict): 包含路径信息的字典。

    Returns:
        seq_dict (dict): 各链的序列。
        index2resid_dict (dict): 各链的残基索引。
    """
    pdb_id = pair.split('_')[0]  # 获取 pdbid
    seq_dict = {}
    index2resid_dict = {}

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_id, protonated_pdb_file)
    model = struct[0]  # 获取第一个模型

    # 获取所有链 ID
    chain_ids = [chain.get_id() for chain in model.get_chains()]

    for chain_id in chain_ids:
        chain = model[chain_id]
        res_type = ''
        index2resid = {}
        for index, res in enumerate(chain):
            resname = res.resname
            res_type += three_to_one.get(resname, 'X')  # 使用字典将三字母代码转换为一字母代码
            index2resid[index] = res.id[1]
        seq_dict[chain_id] = res_type
        index2resid_dict[chain_id] = index2resid

    return seq_dict, index2resid_dict


def extractPDB(pdb_file, pair, dir_opts={}):
    """
    提取 PDB 文件中的相关信息，返回残基信息。

    Args:
        pdb_file (str): PDB 文件路径。
        pair (str): 由蛋白质链信息组成的字符串，如 '1a0q_HL'。
        dir_opts (dict): 包含路径信息的字典。

    Returns:
        residue_info (dict): 残基信息，键为链+残基号，值为残基类型。
    """
    pdb_id = pair.split('_')[0]  # 获取 pdbid
    residue_info = {}

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_id, pdb_file)
    model = struct[0]  # 获取第一个模型

    # 获取所有链 ID
    chain_ids = [chain.get_id() for chain in model.get_chains()]

    for chain_id in chain_ids:
        chain = model[chain_id]
        for residue in chain:
            if residue.get_resname().upper() in Amino_acid_type:
                res_id = "{}_{}".format(chain_id, residue.id[1])
                residue_info[res_id] = residue.get_resname().upper()

    return residue_info


def process_pdb_files(pdb_dir, output_dir):
    """
    处理指定目录中的所有 PDB 文件，生成特征并保存在 pickle 文件中。

    Args:
        pdb_dir (str): PDB 文件所在目录。
        output_dir (str): 输出文件目录。
    """
    dir_opts = {
        'protonated_pdb_dir': pdb_dir,
        'xyzrn_dir': os.path.join(output_dir, 'xyzrn'),
        'raw_pdb_dir': pdb_dir
    }

    xyzrn_data = {}
    seq_data = {}
    residue_data = {}

    for pdb_file in os.listdir(pdb_dir):
        if pdb_file.endswith('.pdb'):
            # 获取pdbid
            pdb_id = os.path.splitext(pdb_file)[0]
            pair = pdb_id  # 这里直接使用pdbid，假设每个文件夹只有一个链
            pdb_path = os.path.join(pdb_dir, pdb_file)

            # 质子化处理
            protonated_pdb_file = protonate(pdb_path, dir_opts)

            # 提取 XYZRN 特征
            extract_xyzrn(protonated_pdb_file, pair, dir_opts)
            with open(os.path.join(dir_opts['xyzrn_dir'], pair + '.xyzrn'), 'r') as f:
                xyzrn_data[pair] = f.readlines()

            # 获取蛋白质序列
            seq_dict, index2resid_dict = get_seq(protonated_pdb_file, pair, dir_opts)
            seq_data[pair] = {
                'seq': seq_dict,
                'index2resid': index2resid_dict
            }

            # 提取 PDB 残基信息
            residue_info = extractPDB(protonated_pdb_file, pair, dir_opts)
            residue_data[pair] = residue_info

    # 保存数据到 pickle 文件
    with open(os.path.join(output_dir, 'xyzrn_data.pkl'), 'wb') as f:
        pickle.dump(xyzrn_data, f)

    with open(os.path.join(output_dir, 'seq_data.pkl'), 'wb') as f:
        pickle.dump(seq_data, f)

    with open(os.path.join(output_dir, 'residue_data.pkl'), 'wb') as f:
        pickle.dump(residue_data, f)


# 示例使用
pdb_dir = './Data/dti/biosnap_pdb'    # or ./Data/pdb_files/protein_pdb  or ./Data/pdb_files/protein_pdb
output_dir = './data_process/mesh/biosnap_pro'
process_pdb_files(pdb_dir, output_dir)