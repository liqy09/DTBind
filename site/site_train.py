import os
import sys
import warnings


warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(''))


import pickle
import numpy as np
import subprocess
import random
from itertools import repeat, product
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import pandas as pd
import argparse
from datetime import datetime
from site_model import *
from site_metric import *
from site_utils import *
from torch.optim import Adam
import sklearn.metrics as skm
import torch.nn.init as init
from sklearn.metrics import roc_curve, auc, confusion_matrix, matthews_corrcoef, precision_recall_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.autograd.set_detect_anomaly(True)


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logging(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(output_folder, f"training_{timestamp}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    print(f"Logging to {log_file}")
    return log_file


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)


def best_f_1(label,output):
    f_1_max = 0
    t_max = 0
    output = np.array(output)
    for t in range(1,100):
        threshold = t / 100
        predict = np.where(output>threshold,1,0)
        f_1 = skm.f1_score(label, predict, pos_label=1)
        if f_1 > f_1_max:
            f_1_max = f_1
            t_max = threshold

    pred = np.where(output>t_max,1,0)
    accuracy = skm.accuracy_score(label, pred)
    recall = skm.recall_score(label, pred)
    precision = skm.precision_score(label, pred)
    MCC = skm.matthews_corrcoef(label, pred)
    return accuracy,recall,precision,MCC,f_1_max,t_max


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []

    for ii, (protein_data, drug_data) in enumerate(train_loader):
        protein_data = protein_data.to(device)
        drug_data = drug_data.to(device)
        optimizer.zero_grad()

        # 假设模型的 forward 方法接受蛋白质数据和药物数据
        out = model(protein_data, drug_data)
        loss = criterion(out.squeeze(-1), protein_data.y)  # 假设标签在蛋白质数据中
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        y_true.extend(protein_data.y.cpu().detach().numpy())
        y_pred.extend(out.cpu().detach().numpy())

        del protein_data, drug_data, out, loss
        torch.cuda.empty_cache()

    metrics = eval_metrics(np.array(y_pred), np.array(y_true))

    del y_true, y_pred
    torch.cuda.empty_cache()

    return {
        'total_loss': total_loss,
        'avg_loss': total_loss / len(train_loader),
        'threshold': metrics['threshold'],
        'acc': metrics['acc'],
        'rec': metrics['rec'],
        'pre': metrics['pre'],
        'f1': metrics['f1'],
        'spe': metrics['spe'],
        'mcc': metrics['mcc'],
        'auc': metrics['auc'],
        'auprc': metrics['auprc']
    }


def validate(model, device, val_loader, criterion, best_threshold=None):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for ii, (protein_data, drug_data) in enumerate(val_loader):
            protein_data = protein_data.to(device)
            drug_data = drug_data.to(device)
            out = model(protein_data, drug_data)
            loss = criterion(out.squeeze(-1), protein_data.y)
            total_loss += loss.item()
            y_true.extend(protein_data.y.cpu().numpy())
            y_pred.extend(out.cpu().numpy())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    if best_threshold is not None:
        metrics = th_eval_metrics(best_threshold, y_pred_np, y_true_np, cal_AUC=True)
    else:
        metrics = eval_metrics(y_pred_np, y_true_np, cal_AUC=True)

    return {
        'total_loss': total_loss,
        'avg_loss': total_loss / len(val_loader),
        'threshold': metrics['threshold'],
        'acc': metrics['acc'],
        'rec': metrics['rec'],
        'pre': metrics['pre'],
        'f1': metrics['f1'],
        'spe': metrics['spe'],
        'mcc': metrics['mcc'],
        'auc': metrics['auc'],
        'auprc': metrics['auprc']
    }


def main(opt, device, params):
    log_file = setup_logging(opt.log_folder)

    # 加载数据集
    train_dataset = ProteinDrugDataset(
        protein_folder=opt.protein_folder,
        drug_folder=opt.drug_folder,
        mapping_file=opt.mapping_file,
        pdb_ids_file=opt.train_ids_file
    )
    val_dataset = ProteinDrugDataset(
        protein_folder=opt.protein_folder,
        drug_folder=opt.drug_folder,
        mapping_file=opt.mapping_file,
        pdb_ids_file=opt.val_ids_file
    )
    test_dataset = ProteinDrugDataset(
        protein_folder=opt.protein_folder,
        drug_folder=opt.drug_folder,
        mapping_file=opt.mapping_file,
        pdb_ids_file=opt.test_ids_file
    )

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True)

    logging.info("Train loader length: %d", len(train_loader))
    logging.info("Valid loader length: %d", len(val_loader))
    logging.info("Test loader length: %d", len(test_loader))

    print("Train loader length:", len(train_loader))
    print("Valid loader length:", len(val_loader))
    print("Test loader length:", len(test_loader))

    model = DTISite(params)
    model.apply(init_weights)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")

    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    criterion = FocalLoss(alpha=0.25, gamma=2, logits=False, reduction='mean')  # 使用 FocalLoss

    best_metrics = {"f1": -1.0, "mcc": -1.0, "auc": -1.0}
    best_threshold = 0.4

    for epoch in range(opt.epochs):
        train_metrics = train(model, device, train_loader, optimizer, criterion)
        val_metrics = validate(model, device, val_loader, criterion)
        # test_metrics = validate(model, device, test_loader, criterion, best_threshold=val_metrics['threshold'])
        test_metrics = validate(model, device, test_loader, criterion)

        # logging.info(f"Epoch {epoch + 1}/{opt.epochs}")
        # logging.info(
        #     f"Train Loss: {train_metrics['total_loss']:.4f}, avg loss: {train_metrics['avg_loss']:.4f}, AUC: {train_metrics['auc']:.4f}, MCC: {train_metrics['mcc']:.4f}, F1: {train_metrics['f1']:.4f}, auprc:{train_metrics['auprc']:.4f}, rec: {train_metrics['rec']:.4f}, pre: {train_metrics['pre']:.4f}, acc:{train_metrics['acc']:.4f}, spe: {train_metrics['spe']:.4f}, mcc: {train_metrics['mcc']:.4f}")
        # logging.info(
        #     f"Val Loss: {val_metrics['total_loss']:.4f}, avg loss: {val_metrics['avg_loss']:.4f}, AUC: {val_metrics['auc']:.4f}, MCC: {val_metrics['mcc']:.4f}, F1: {val_metrics['f1']:.4f}, auprc:{val_metrics['auprc']:.4f}, rec: {val_metrics['rec']:.4f}, pre: {val_metrics['pre']:.4f}, acc:{val_metrics['acc']:.4f}, spe: {val_metrics['spe']:.4f}, threshold:{val_metrics['threshold']}")
        # logging.info(
        #     f"Test Loss: {test_metrics['total_loss']:.4f}, avg loss: {test_metrics['avg_loss']:.4f}, AUC: {test_metrics['auc']:.4f}, MCC: {test_metrics['mcc']:.4f}, F1: {test_metrics['f1']:.4f}, auprc:{test_metrics['auprc']:.4f}, rec: {test_metrics['rec']:.4f}, pre: {test_metrics['pre']:.4f}, acc:{test_metrics['acc']:.4f}, spe: {test_metrics['spe']:.4f}, threshold:{test_metrics['threshold']}")
        #
        # print(f"Epoch {epoch + 1}/{opt.epochs}")
        # print(
        #     f"Train Loss: {train_metrics['total_loss']:.4f}, avg loss: {train_metrics['avg_loss']:.4f}, AUC: {train_metrics['auc']:.4f}, MCC: {train_metrics['mcc']:.4f}, F1: {train_metrics['f1']:.4f}, auprc:{train_metrics['auprc']:.4f}, rec: {train_metrics['rec']:.4f}, pre: {train_metrics['pre']:.4f}, acc:{train_metrics['acc']:.4f}, spe: {train_metrics['spe']:.4f}, mcc: {train_metrics['mcc']:.4f}")
        # print(
        #     f"Val Loss: {val_metrics['total_loss']:.4f}, avg loss: {val_metrics['avg_loss']:.4f}, AUC: {val_metrics['auc']:.4f}, MCC: {val_metrics['mcc']:.4f}, F1: {val_metrics['f1']:.4f}, auprc:{val_metrics['auprc']:.4f}, rec: {val_metrics['rec']:.4f}, pre: {val_metrics['pre']:.4f}, acc:{val_metrics['acc']:.4f}, spe: {val_metrics['spe']:.4f}, threshold:{val_metrics['threshold']}")
        # print(
        #     f"Test Loss: {test_metrics['total_loss']:.4f}, avg loss: {test_metrics['avg_loss']:.4f}, AUC: {test_metrics['auc']:.4f}, MCC: {test_metrics['mcc']:.4f}, F1: {test_metrics['f1']:.4f}, auprc:{test_metrics['auprc']:.4f}, rec: {test_metrics['rec']:.4f}, pre: {test_metrics['pre']:.4f}, acc:{test_metrics['acc']:.4f}, spe: {test_metrics['spe']:.4f}, threshold:{test_metrics['threshold']}")

        if val_metrics['f1'] > best_metrics["f1"] or (val_metrics['f1'] == best_metrics["f1"] and val_metrics['mcc'] > best_metrics["mcc"]) or (
                val_metrics['f1'] == best_metrics["f1"] and val_metrics['mcc'] == best_metrics["mcc"] and val_metrics['auc'] > best_metrics["auc"]):
            best_metrics = {"f1": val_metrics['f1'], "mcc": val_metrics['mcc'], "auc": val_metrics['auc']}
            best_threshold = val_metrics['threshold']
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': best_threshold
            }, opt.model_path)

            logging.info(
                f"Model saved at Epoch {epoch + 1} with Val F1: {val_metrics['f1']:.4f}, MCC: {val_metrics['mcc']:.4f}, AUC: {val_metrics['auc']:.4f}")
            print(f"Model saved at Epoch {epoch + 1} with Val F1: {val_metrics['f1']:.4f}, MCC: {val_metrics['mcc']:.4f}, AUC: {val_metrics['auc']:.4f}")

        scheduler.step(val_metrics['f1'])

    test_metrics = validate(model, device, test_loader, criterion, best_threshold=best_threshold)
    logging.info(
        f"Final Test Loss: {test_metrics['total_loss']:.4f}, AUC: {test_metrics['auc']:.4f}, MCC: {test_metrics['mcc']:.4f}, F1: {test_metrics['f1']:.4f}, auprc:{test_metrics['auprc']:.4f}, rec: {test_metrics['rec']:.4f}, pre: {test_metrics['pre']:.4f}, acc:{test_metrics['acc']:.4f}, spe: {test_metrics['spe']:.4f}, thre:{test_metrics['threshold']}")
    print(
        f"Final Test Loss: {test_metrics['total_loss']:.4f}, AUC: {test_metrics['auc']:.4f}, MCC: {test_metrics['mcc']:.4f}, F1: {test_metrics['f1']:.4f}, auprc:{test_metrics['auprc']:.4f}, rec: {test_metrics['rec']:.4f}, pre: {test_metrics['pre']:.4f}, acc:{test_metrics['acc']:.4f}, spe: {test_metrics['spe']:.4f}, thre:{test_metrics['threshold']}")



if __name__ == "__main__":
    # 设置参数
    setup_seed(42)

    class Options:
        def __init__(self):

            self.protein_folder = "./Data/site/protein_graph"
            self.drug_folder = "./Data/site/ligand_graph"
            self.mapping_file = "./Data/site/pl_map.txt"
            self.train_ids_file = "./Data/site/train_ids.txt"
            self.val_ids_file = "./Data/site/val_ids.txt"
            self.test_ids_file = "./Data/site/test_ids.txt"
            self.model_path = "./models/site_model.pth"
            self.log_folder = "logs/site"
            self.batch_size = 16
            self.lr = 0.0005
            self.weight_decay = 1e-5
            self.epochs =  70


    opt = Options()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(opt, device, params)


