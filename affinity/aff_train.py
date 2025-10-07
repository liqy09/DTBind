import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import gc
from torchnet import meter
import psutil
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from metrics import rmse, pearsonr, spearmanr
from scipy.stats import pearsonr, spearmanr
import importlib
import matplotlib.pyplot as plt
from aff_model import *
from aff_model import *
import os
torch.autograd.set_detect_anomaly(True)


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.__version__)
print(torch.cuda.is_available())


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")  # 以MB为单位显示

def collate_fn(batch):
    if len(batch[0]) != 24:
        raise ValueError(f"Expected 10 elements in each batch, but got {len(batch[0])}. Please check the dataset.")

    pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs, p_num_nbs_mar, \
        a_fea, b_fea, d_anb, d_bnb, d_nbs, num_nbs_mar, affinity_labels, \
        edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, \
        d_bond_nei, p_nei_nbs, d_nei_nbs, p_nbs_mar, d_nbs_mar = zip(*batch)

    # 返回填充后的标签和其他数据
    return (list(pretrained_fea), list(surface_fea), list(p_edge_fea), list(p_anb), list(p_bnb), list(p_nbs), list(p_num_nbs_mar), list(a_fea), list(b_fea),
            list(d_anb), list(d_bnb), list(d_nbs), list(num_nbs_mar), list(affinity_labels), list(edge_residue_to_atom), list(edge_atom_to_residue), list(p_nei), list(d_nei), list(p_bond_nei), list(d_bond_nei),
            list(p_nei_nbs), list(d_nei_nbs), list(p_nbs_mar), list(d_nbs_mar))


def train_affinity(train_data, valid_data, test_data, params, batch_size, n_epoch, patience=25, min_delta=0.001):
    best_valid_score = float('inf')
    epochs_no_improve = 0
    criterion = nn.MSELoss()
    best_test_scores = None
    test_scores_history = []
    valid_scores_history = []
    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []
    train_pearson_history = []
    valid_pearson_history = []
    test_pearson_history = []
    test_scores_history = []
    valid_scores_history = []

    # 初始化模型
    model = DTIModel(params)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")

    loss_meter = meter.AverageValueMeter()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=0.01, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, min_lr=1e-6)

    train_loader = DataLoader(list(zip(*train_data)), batch_size=batch_size, shuffle=True,  drop_last=False, collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(list(zip(*valid_data)), batch_size=batch_size, shuffle=False,  drop_last=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(list(zip(*test_data)), batch_size=batch_size, shuffle=False,  drop_last=False, collate_fn=collate_fn, num_workers=0)

    print("train loader length:", len(train_loader))
    print("valid loader length:", len(valid_loader))
    print("Test loader length:", len(test_loader))

    # 训练循环
    for epoch in range(n_epoch):
        # print(f"Epoch {epoch + 1}/{n_epoch}:")
        train_loss, train_rmse, train_pearson_corr, spearman_corr, r2, mae = train_a_epoch(model, train_loader, optimizer, criterion)
        # print(f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, pearson: {train_pearson_corr:.4f}, spearman: {spearman_corr:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
        train_loss_history.append(train_loss)  # 记录训练损失
        train_pearson_history.append(train_pearson_corr)  # 记录训练皮尔森相关系数

        valid_loss, valid_mse, valid_pearson, valid_spearman, r2, mae, valid_preds, valid_labels = test_a_epoch(model, valid_loader, criterion, device)
        valid_scores_history.append((valid_mse, valid_pearson, valid_spearman, r2, mae))
        valid_loss_history.append(valid_loss)  # 记录验证损失
        valid_pearson_history.append(valid_pearson)  # 记录验证皮尔森相关系数
        # print(f"Validation Performance at Epoch {epoch + 1}: {valid_mse}")
        # print(f"Epoch {epoch + 1} - Valid loss: {valid_loss:.4f}, RMSE: {valid_mse:.4f}, Pearson: {valid_pearson:.4f}, Spearman: {valid_spearman:.4f}, R2:{r2:.4f}, MAE:{mae:.4f}")

        # print(f"Training Loss: {train_loss:.4f} | Validation MSE: {valid_mse:.4f}")

        scheduler.step(valid_mse)
        loss_meter.reset()

        del valid_preds, valid_labels
        gc.collect()
        torch.cuda.empty_cache()

        if valid_mse < best_valid_score - min_delta:
            best_valid_score = valid_mse
            epochs_no_improve = 0

            avg_loss, test_rmse, test_pearson, test_spearman, r2, mae, test_preds, test_labels = test_a_epoch(model, test_loader, criterion, device)
            # print(f"Epoch {epoch + 1} - Test loss: {avg_loss:.4f}, RMSE: {test_rmse:.4f}, Pearson: {test_pearson:.4f}, Spearman: {test_spearman:.4f}, R2:{r2:.4f}, MAE:{mae:.4f}")
            best_test_scores = (test_rmse, test_pearson, test_spearman, r2, mae)
            test_scores_history.append((test_rmse, test_pearson, test_spearman, r2, mae))
            test_loss_history.append(avg_loss)  # 记录测试损失
            test_pearson_history.append(test_pearson)  # 记录测试皮尔森相关系数
            # print(f"Epoch {epoch + 1} - Best Test loss: {avg_loss:.4f}, Test RMSE: {test_rmse:.4f}, Pearson: {test_pearson:.4f}, Spearman: {test_spearman:.4f}, R2:{r2:.4f}, MAE:{mae:.4f}")

            # 保存最佳模
            del test_preds, test_labels
            gc.collect()
            torch.cuda.empty_cache()

            torch.save(model.state_dict(), "models/model_affinity.pth")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        # 提前停止
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    print("Training Finished.")
    print(f"Best Validation MSE: {best_valid_score}")


    if best_test_scores:
        best_test_rmse, best_test_pearson, best_test_spearman, best_test_r2, best_test_mae = best_test_scores
        print(f"Best Test Scores corresponding to Best Validation MSE:")
        print(
            f"Test RMSE: {best_test_rmse:.4f}, Pearson: {best_test_pearson:.4f}, Spearman: {best_test_spearman:.4f}, R2: {best_test_r2:.4f}, MAE: {best_test_mae:.4f}")


    return best_valid_score, best_test_scores, avg_valid_mse, avg_test_rmse, avg_test_pearson, avg_test_spearman, avg_test_r2, avg_test_mae

def train_a_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    preds = []
    all_labels = []
    for batch_data in train_loader:

        (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs, p_num_nbs_mar,
            a_fea, b_fea, d_anb, d_bnb, d_nbs, num_nbs_mar, affinity_labels,
            edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei,
            d_bond_nei, p_nei_nbs, d_nei_nbs, p_nbs_mar, d_nbs_mar) = batch_data

        (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs_mask, p_mask, a_fea, b_fea, d_anb, d_bnb, d_nbs_mask, d_mask, affinity_labels,
            edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, d_bond_nei, p_hetero_mask, d_hetero_mask) = batch_data_process(batch_data)


        affinity_labels = torch.tensor(affinity_labels, dtype=torch.float32)
        labels = affinity_labels.to(device)
        # print(affinity_labels.shape)

        # 梯度清零
        optimizer.zero_grad()

        affinity_pred = model(pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs_mask, p_mask, a_fea, b_fea, d_anb, d_bnb, d_nbs_mask, d_mask,
                              edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, d_bond_nei, p_hetero_mask, d_hetero_mask)

        # 计算损失
        loss = criterion(affinity_pred.squeeze(), labels)
        # loss = criterion(torch.cat(affinity_pred).squeeze(), labels)
        # print(loss.item())
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()
        preds.extend([pred.cpu().detach().numpy() for pred in affinity_pred])
        all_labels.extend([label.cpu().detach().numpy() for label in labels])

        # 将预测和标签列表转换为NumPy数组
    if preds and all_labels:  # 确保列表不为空
        # 确保每个元素都是一维数组
        preds = [pred.flatten() for pred in preds]
        all_labels = [label.flatten() for label in all_labels]
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
    else:
        preds = np.array([])
        labels = np.array([])

    avg_loss = total_loss / len(train_loader)

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(labels, preds))
    pearson_corr, _ = pearsonr(labels, preds)
    spearman_corr, _ = spearmanr(labels, preds)
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)

    # 返回平均训练损失
    return avg_loss, rmse, pearson_corr, spearman_corr, r2, mae


def test_a_epoch(model, test_loader, criterion, device):

    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    for batch_data in test_loader:
        (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs, p_num_nbs_mar,
         a_fea, b_fea, d_anb, d_bnb, d_nbs, num_nbs_mar, affinity_labels,
         edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei,
         d_bond_nei, p_nei_nbs, d_nei_nbs, p_nbs_mar, d_nbs_mar) = batch_data

        (pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs_mask, p_mask, a_fea, b_fea, d_anb, d_bnb, d_nbs_mask, d_mask, affinity_labels,
            edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, d_bond_nei, p_hetero_mask, d_hetero_mask) = batch_data_process(batch_data)


        affinity_labels = torch.tensor(affinity_labels, dtype=torch.float32)
        labels = affinity_labels.to(device)

        affinity_pred = model(pretrained_fea, surface_fea, p_edge_fea, p_anb, p_bnb, p_nbs_mask, p_mask, a_fea, b_fea, d_anb, d_bnb, d_nbs_mask, d_mask,
                              edge_residue_to_atom, edge_atom_to_residue, p_nei, d_nei, p_bond_nei, d_bond_nei, p_hetero_mask, d_hetero_mask)

        # 计算损失
        loss = criterion(affinity_pred.squeeze(), labels)
        total_loss += loss.item()

        all_preds.extend([pred.cpu().detach().numpy() for pred in affinity_pred])
        all_labels.extend([label.cpu().detach().numpy() for label in labels])

    if all_preds and all_labels:
        all_preds = [pred.flatten() for pred in all_preds]
        all_labels = [label.flatten() for label in all_labels]
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
    else:
        preds = np.array([])
        labels = np.array([])

    avg_loss = total_loss / len(test_loader)

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(labels, preds))
    pearson_corr, _ = pearsonr(labels, preds)
    spearman_corr, _ = spearmanr(labels, preds)
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)

    # print(f"Test performance: RMSE={rmse:.4f}, Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}, R2={r2:.4f}, MAE={mae:.4f}")

    return avg_loss, rmse, pearson_corr, spearman_corr, r2, mae, preds, labels


if __name__ == "__main__":
    setup_seed(9)

    TRAIN_PARAMS = {
        'batch_size': 32,
        'n_epoch': 60,
        'learning_rate': 0.0005,
    }

    train_data_path = './data/affinity/train_graph.pkl'
    valid_data_path = './data/affinity/val_graph.pkl'
    test_data_path = './data/affinity/test_graph.pkl'


    train_data = loading_emb(train_data_path)
    valid_data = loading_emb(valid_data_path)
    test_data = loading_emb(test_data_path)

    # 调用数据分割函数
    train_data, valid_data, test_data = split_train_valid_test(train_data, valid_data, test_data)


    # 训练模型
    best_valid_score, best_test_scores, avg_valid_mse, avg_test_rmse, avg_test_pearson, avg_test_spearman, avg_test_r2, avg_test_mae = train_affinity(
        train_data, valid_data, test_data,
        params, TRAIN_PARAMS['batch_size'], TRAIN_PARAMS['n_epoch'],
        patience=20, min_delta=0.001
    )

    print('Training completed.')
