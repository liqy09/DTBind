import os
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, DataLoader
import logging
import numpy as np
from datetime import datetime
from site_model import *  # 确保模型类名与训练代码一致
import pandas as pd

def setup_logging(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(output_folder, f"testing_{timestamp}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    print(f"Logging to {log_file}")
    return log_file

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    best_threshold = checkpoint['best_threshold']

    model = DTISite(params)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    return model, best_threshold

class ProteinDrugDataset(Dataset):
    def __init__(self, protein_folder, drug_folder, mapping_file, pdb_ids_file):
        self.protein_folder = protein_folder
        self.drug_folder = drug_folder
        self.mapping = self.load_mapping(mapping_file)
        self.pdb_ids = self.load_pdb_ids(pdb_ids_file)

        existing_protein_files = {os.path.splitext(file)[0] for file in os.listdir(self.protein_folder) if file.endswith(".pt")}
        existing_drug_files = {os.path.splitext(file)[0] for file in os.listdir(self.drug_folder) if file.endswith(".pt")}

        self.pdb_ids = [
            pdb_id for pdb_id in self.pdb_ids
            if pdb_id in existing_protein_files and
               (pdb_id in self.mapping and self.mapping[pdb_id] in existing_drug_files)
        ]

        # 添加 valid_mappings 属性
        self.valid_mappings = [(pdb_id, self.mapping[pdb_id]) for pdb_id in self.pdb_ids]

    def load_mapping(self, mapping_file):
        mapping = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                pdb_id, drug_id = line.strip().split()
                mapping[pdb_id] = drug_id
        return mapping

    def load_pdb_ids(self, pdb_ids_file):
        with open(pdb_ids_file, 'r') as f:
            pdb_ids = [line.strip() for line in f]
        return pdb_ids

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        drug_id = self.mapping[pdb_id]

        protein_file = os.path.join(self.protein_folder, f"{pdb_id}.pt")
        protein_data = torch.load(protein_file)

        drug_file = os.path.join(self.drug_folder, f"{drug_id}.pt")
        drug_data = torch.load(drug_file)

        return protein_data, drug_data


def test(model, device, test_loader):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for protein_data, drug_data in test_loader:
            protein_data = protein_data.to(device)
            drug_data = drug_data.to(device)
            out = model(protein_data, drug_data)
            
            y_pred.append(out.cpu().numpy())

    return y_pred

def main(opt, device):
    log_file = setup_logging(opt.log_folder)

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    test_dataset = ProteinDrugDataset(
        protein_folder=opt.protein_folder,
        drug_folder=opt.drug_folder,
        mapping_file=opt.mapping_file,
        pdb_ids_file=opt.test_ids_file
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)  # 设置 batch_size 为 1

    logging.info("Test loader length: %d", len(test_loader))
    print("Test loader length:", len(test_loader))

    model, best_threshold = load_model(opt.model_path, device)

    y_pred_list = test(model, device, test_loader)

    for idx, (pdb_id, drug_id) in enumerate(test_dataset.valid_mappings):
        predictions = y_pred_list[idx].flatten()
        binary_predictions = (predictions > best_threshold).astype(int)

        output_csv = os.path.join(opt.output_folder, f"{pdb_id}_{drug_id}.csv")
        df = pd.DataFrame({
            "Prediction": predictions,
            "Binary_Prediction": binary_predictions
        })
        df.to_csv(output_csv, index=False)
        logging.info(f"Saved prediction for {pdb_id}_{drug_id} to {output_csv}")
        print(f"Saved prediction for {pdb_id}_{drug_id} to {output_csv}")

if __name__ == "__main__":
    class Options:
        def __init__(self):
            self.protein_folder = "../../sample_test/site/protein_graph"
            self.drug_folder = "../../sample_test/site/ligand_graph"
            self.mapping_file = "../../sample_test/site/sample_map.txt"
            self.test_ids_file = "../../sample_test/site/sample_id.txt"
            self.model_path = "../../models/site_model.pth"
            self.log_folder = "../../logs/site"
            self.output_folder = "../../logs/site/sample_predictions"  # 指定保存预测结果的文件夹
            self.batch_size = 1

    opt = Options()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(opt, device)
