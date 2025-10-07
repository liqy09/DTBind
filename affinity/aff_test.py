import torch
import numpy as np
import os
import pickle
import pandas as pd
from aff_model import DTIModel  # 假设这是你训练的模型
from aff_utils import *  # 你的一些工具函数
import torch
from torch.autograd import Variable
import numpy as np
import os
import pandas as pd
import pickle

# 设备配置
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

model = DTIModel(params)
model.load_state_dict(torch.load("models/affinity_model.pth", map_location=device))
model = model.to(device)
model.eval()


def process_single_sample(sample_data):

    (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs, p_num_nbs_mar,
     a_fea, b_fea, d_anb, d_bnb, d_nbs, num_nbs_mar, affinity_label,
     edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei,
     d_bond_nei, p_nei_nbs, d_nei_nbs, p_nbs_mar, d_nbs_mar) = sample_data

    pretrained_fea = [pretrained_fea]
    surface_fea = [surface_fea]
    p_edge_fea = [p_edge_fea]
    p_anb = [p_anb]
    p_bnb = [p_bnb]
    p_nbs = [p_nbs]
    p_num_nbs_mar = [p_num_nbs_mar]
    a_fea = [a_fea]
    b_fea = [b_fea]
    d_anb = [d_anb]
    d_bnb = [d_bnb]
    d_nbs = [d_nbs]
    num_nbs_mar = [num_nbs_mar]
    affinity_labels = [affinity_label]
    edge_residue_to_atom = [edge_residue_to_atom]
    edge_atom_to_residue = [edge_atom_to_residue]
    p_nei = [p_nei]
    d_nei = [d_nei]
    p_bond_nei = [p_bond_nei]
    d_bond_nei = [d_bond_nei]
    p_nei_nbs = [p_nei_nbs]
    d_nei_nbs = [d_nei_nbs]
    p_nbs_mar = [p_nbs_mar]
    d_nbs_mar = [d_nbs_mar]

    (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs_mask, p_mask,
     a_fea, b_fea, d_anb, d_bnb, d_nbs_mask, d_mask, affinity_labels,
     edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei,
     d_bond_nei, p_hetero_mask, d_hetero_mask) = batch_data_process(
        (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs, p_num_nbs_mar,
         a_fea, b_fea, d_anb, d_bnb, d_nbs, num_nbs_mar, affinity_labels,
         edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei,
         d_bond_nei, p_nei_nbs, d_nei_nbs, p_nbs_mar, d_nbs_mar)
    )

    return (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs_mask, p_mask,
            a_fea, b_fea, d_anb, d_bnb, d_nbs_mask, d_mask,
            edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei,
            d_bond_nei, p_hetero_mask, d_hetero_mask, affinity_labels[0])

def predict_single_sample(sample_path):

    with open(sample_path, 'rb') as f:
        sample_data = pickle.load(f)

    processed_data = process_single_sample(sample_data)

    inputs = processed_data[:-1]
    true_value = processed_data[-1]

    with torch.no_grad():
        pred_tensor = model(*inputs)
        pred_value = pred_tensor.item()

    error = abs(true_value - pred_value)

    pdbid = os.path.basename(sample_path).split('.')[0]

    return pdbid, true_value, pred_value, error

def process_all_samples(data_dir, output_csv="sample_aff_results.csv"):
    results = []

    # 获取所有pkl文件
    sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    total = len(sample_files)

    print(f"开始处理 {total} 个样本...")

    for i, filename in enumerate(sample_files):
        sample_path = os.path.join(data_dir, filename)

        try:
            pdbid, true_value, pred_value, error = predict_single_sample(sample_path)
            results.append({
                'pdbid': pdbid,
                'true_value': true_value,
                'pred_value': pred_value,
                'error': error
            })
            print(
                f"处理进度: {i + 1}/{total} - {pdbid}: 真实值={true_value:.4f}, 预测值={pred_value:.4f}, 误差={error:.4f}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='error')

    df_sorted.to_csv(output_csv, index=False)
    print(f"结果已保存到 {output_csv}")

    return df_sorted

if __name__ == "__main__":

    data_dir = "/mnt/storage1/liqy/DTBind-main/sample_test/affinity"  # 替换为你的单个样本目录
    output_csv = "sample_results.csv"

    results_df = process_all_samples(data_dir, output_csv)

    print("\n预测结果（按误差排序）:")
    print(results_df)