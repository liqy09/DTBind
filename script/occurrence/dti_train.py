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
from dti_model import *
from dti_utils import *
from torch.optim import Adam
import sklearn.metrics as skm
import torch.nn.init as init
from sklearn.metrics import roc_curve, auc, confusion_matrix, matthews_corrcoef, precision_recall_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.autograd.set_detect_anomaly(True)



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.constant_(m.in_proj_bias, 0)
        nn.init.xavier_uniform_(m.out_proj.weight)
        nn.init.constant_(m.out_proj.bias, 0)


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

    for ii, (protein_data, drug_data, label) in enumerate(train_loader):

        # print(type(label))
        # print(f"label:{label}")

        protein_data = protein_data.to(device)
        drug_data = drug_data.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        # 假设模型的 forward 方法接受蛋白质数据和药物数据
        out = model(protein_data, drug_data)
        # print(type(out))
        # print(f"out:{out}")
        loss = criterion(out.squeeze(-1), label.float())  # 假设标签在蛋白质数据中
        # print(f"loss:{loss}")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        y_true.extend(label.cpu().detach().numpy())
        y_pred.extend(out.cpu().detach().numpy())

        del protein_data, drug_data, out, loss
        torch.cuda.empty_cache()

    metrics = eval_metrics(np.array(y_pred), np.array(y_true))
    # threshold, acc, rec, pre, F1, spe, mcc, auc_, auprc, _ = eval_metrics(np.array(y_pred), np.array(y_true))

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
        for ii, (protein_data, drug_data, label) in enumerate(val_loader):
            protein_data = protein_data.to(device)
            drug_data = drug_data.to(device)
            label = label.to(device)
            out = model(protein_data, drug_data)
            loss = criterion(out.squeeze(-1), label.float())
            total_loss += loss.item()
            y_true.extend(label.cpu().numpy())
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
        tsv_file=opt.train_tsv_file)

    val_dataset = ProteinDrugDataset(
        protein_folder=opt.protein_folder,
        drug_folder=opt.drug_folder,
        tsv_file=opt.val_tsv_file)

    test_dataset = ProteinDrugDataset(
        protein_folder=opt.protein_folder,
        drug_folder=opt.drug_folder,
        tsv_file=opt.test_tsv_file)


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
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=7, min_lr=1e-6)
    criterion = nn.BCELoss()

    # criterion = FocalLoss(alpha=0.25, gamma=2, logits=False, reduction='mean')  # 使用 FocalLoss

    best_metrics = {"f1": -1.0, "mcc": -1.0, "auc": -1.0}
    best_threshold = 0.4

    for epoch in range(opt.epochs):

        train_metrics = train(
            model, device, train_loader, optimizer, criterion)
        total_loss, train_loss, train_auc, train_mcc, train_f1, train_rec, train_pre, train_spe, train_auprc, train_acc, train_threshold = (
            train_metrics['total_loss'], train_metrics['avg_loss'], train_metrics['auc'], train_metrics['mcc'], train_metrics['f1'],
            train_metrics['rec'], train_metrics['pre'], train_metrics['spe'], train_metrics['auprc'], train_metrics['acc'],
            train_metrics['threshold']
        )

        val_metrics = validate(model, device, val_loader, criterion)
        val_loss, avg_loss, val_threshold, val_acc, val_rec, val_pre, val_f1, val_spe, val_mcc, val_auc, val_auprc = (
            val_metrics['total_loss'], val_metrics['avg_loss'], val_metrics['threshold'], val_metrics['acc'],
            val_metrics['rec'], val_metrics['pre'], val_metrics['f1'], val_metrics['spe'], val_metrics['mcc'],
            val_metrics['auc'], val_metrics['auprc']
        )

        test_metrics = validate(model, device, test_loader, criterion, best_threshold=val_threshold)
        test_loss, t_avg_loss, test_threshold, test_acc, test_rec, test_pre, test_f1, test_spe, test_mcc, test_auc, test_auprc = (
            test_metrics['total_loss'], test_metrics['avg_loss'], test_metrics['threshold'], test_metrics['acc'],
            test_metrics['rec'], test_metrics['pre'], test_metrics['f1'], test_metrics['spe'], test_metrics['mcc'],
            test_metrics['auc'], test_metrics['auprc']
        )

        # logging.info(f"Epoch {epoch + 1}/{opt.epochs}")
        # logging.info(
        #     f"Train Loss: {total_loss:.4f}, avg loss: {train_loss:.4f}, AUC: {train_auc:.4f}, MCC: {train_mcc:.4f}, F1: {train_f1:.4f}, auprc:{train_auprc:.4f}, rec: {train_rec:.4f}, pre: {train_pre:.4f}, acc:{train_acc:.4f}, spe: {train_spe:.4f}, mcc: {train_mcc:.4f}")
        # logging.info(
        #     f"Val Loss: {val_loss:.4f}, avg loss: {avg_loss:.4f}, AUC: {val_auc:.4f}, MCC: {val_mcc:.4f}, F1: {val_f1:.4f}, auprc:{val_auprc:.4f}, rec: {val_rec:.4f}, pre: {val_pre:.4f}, acc:{val_acc:.4f}, spe: {val_spe:.4f}, threshold:{val_threshold}")
        # logging.info(
        #     f"Test Loss: {test_loss:.4f}, avg loss: {t_avg_loss:.4f}, AUC: {test_auc:.4f}, MCC: {test_mcc:.4f}, F1: {test_f1:.4f}, auprc:{test_auprc:.4f}, rec: {test_rec:.4f}, pre: {test_pre:.4f}, acc:{test_acc:.4f}, spe: {test_spe:.4f}, threshold:{test_threshold}")
        #
        # print(f"Epoch {epoch + 1}/{opt.epochs}")
        # print(
        #     f"Train Loss: {total_loss:.4f}, avg loss: {train_loss:.4f}, AUC: {train_auc:.4f}, MCC: {train_mcc:.4f}, F1: {train_f1:.4f}, auprc:{train_auprc:.4f}, rec: {train_rec:.4f}, pre: {train_pre:.4f}, acc:{train_acc:.4f}, spe: {train_spe:.4f}, mcc: {train_mcc:.4f}")
        # print(
        #     f"Val Loss: {val_loss:.4f}, avg loss: {avg_loss:.4f}, AUC: {val_auc:.4f}, MCC: {val_mcc:.4f}, F1: {val_f1:.4f}, auprc:{val_auprc:.4f}, rec: {val_rec:.4f}, pre: {val_pre:.4f}, acc:{val_acc:.4f}, spe: {val_spe:.4f}, threshold:{val_threshold}")
        # print(
        #     f"Test Loss: {test_loss:.4f}, avg loss: {t_avg_loss:.4f}, AUC: {test_auc:.4f}, MCC: {test_mcc:.4f}, F1: {test_f1:.4f}, auprc:{test_auprc:.4f}, rec: {test_rec:.4f}, pre: {test_pre:.4f}, acc:{test_acc:.4f}, spe: {test_spe:.4f}, threshold:{test_threshold}")

        if val_f1 > best_metrics["f1"] or (val_f1 == best_metrics["f1"] and val_mcc > best_metrics["mcc"]) or (
                val_f1 == best_metrics["f1"] and val_mcc == best_metrics["mcc"] and val_auc > best_metrics["auc"]):
            best_metrics = {"f1": val_f1, "mcc": val_mcc, "auc": val_auc}
            best_threshold = val_threshold
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': best_threshold
            }, opt.model_path)

            # logging.info(
            #     f"Model saved at Epoch {epoch + 1} with Val F1: {val_f1:.4f}, MCC: {val_mcc:.4f}, AUC: {val_auc:.4f}")
            # print(f"Model saved at Epoch {epoch + 1} with Val F1: {val_f1:.4f}, MCC: {val_mcc:.4f}, AUC: {val_auc:.4f}")

        scheduler.step(val_f1)

    test_metrics = validate(model, device, val_loader, criterion, best_threshold=val_threshold)
    test_loss, t_avg_loss, test_threshold, test_acc, test_rec, test_pre, test_f1, test_spe, test_mcc, test_auc, test_auprc = (
        test_metrics['total_loss'], test_metrics['avg_loss'], test_metrics['threshold'], test_metrics['acc'],
        test_metrics['rec'], test_metrics['pre'], test_metrics['f1'], test_metrics['spe'], test_metrics['mcc'],
        test_metrics['auc'], test_metrics['auprc']
    )

    logging.info(
        f"Final Test Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, MCC: {test_mcc:.4f}, F1: {test_f1:.4f}, auprc:{test_auprc:.4f}, rec: {test_rec:.4f}, pre: {test_pre:.4f}, acc:{test_acc:.4f}, spe: {test_spe:.4f}, thre:{test_threshold}")
    print(
        f"Final Test Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, MCC: {test_mcc:.4f}, F1: {test_f1:.4f}, auprc:{test_auprc:.4f}, rec: {test_rec:.4f}, pre: {test_pre:.4f}, acc:{test_acc:.4f}, spe: {test_spe:.4f}, thre:{test_threshold}")


if __name__ == "__main__":

    setup_seed(42)

    class Options:
        def __init__(self):

            self.protein_folder = "../../Data/dti/protein_graph"
            self.drug_folder = "../../Data/dti/drug_graph"
            self.train_tsv_file = "../../Data/dti/train_label.tsv"
            self.val_tsv_file = "../../Data/dti/val_label.tsv"
            self.test_tsv_file = "../../Data/dti/test_label.tsv"
            self.model_path = "../../models/dti_model.pth"
            self.log_folder = "../../logs/dti"
            self.batch_size = 32
            self.lr = 0.0005
            self.weight_decay = 1e-5
            self.epochs = 70


    opt = Options()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(opt, device, params)


