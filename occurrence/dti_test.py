import os
import logging
import torch
import pandas as pd
from datetime import datetime
from torch_geometric.data import Data, DataLoader
from dti_model import *  # 确保模型定义与训练时一致
from dti_utils import *  # 确保工具函数与训练时一致


def setup_logging(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(output_folder, f"prediction_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")
    return log_file


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model_state_dict = checkpoint['model_state_dict']
    best_threshold = checkpoint.get('best_threshold', 0.5)
    return model_state_dict, best_threshold


class ProteinDrugDataset(Dataset):
    def __init__(self, protein_folder, drug_folder, tsv_file):
        self.protein_folder = protein_folder
        self.drug_folder = drug_folder
        self.data = self.load_tsv(tsv_file)

        existing_protein_files = {os.path.splitext(file)[0] for file in os.listdir(self.protein_folder) if
                                  file.endswith(".pt")}
        existing_drug_files = {os.path.splitext(file)[0] for file in os.listdir(self.drug_folder) if
                               file.endswith(".pt")}

        self.data = [item for item in self.data if
                     item['protein'] in existing_protein_files and item['drug'] in existing_drug_files]

        if not self.data:
            raise ValueError("No valid data found in the dataset.")

    def load_tsv(self, tsv_file):
        data = []
        with open(tsv_file, 'r') as f:
            next(f)  # 跳过表头
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    drug_id, protein_id, label = parts
                    data.append({
                        'drug': drug_id,
                        'protein': protein_id,
                        'label': int(label)
                    })
                else:
                    drug_id, protein_id = parts[0], parts[1]
                    data.append({
                        'drug': drug_id,
                        'protein': protein_id,
                        'label': -1
                    })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        drug_id = item['drug']
        protein_id = item['protein']
        label = item['label']

        protein_file = os.path.join(self.protein_folder, f"{protein_id}.pt")
        protein_data = torch.load(protein_file)

        drug_file = os.path.join(self.drug_folder, f"{drug_id}.pt")
        drug_data = torch.load(drug_file)

        return protein_data, drug_data, label, drug_id, protein_id


def predict(model, device, test_loader, best_threshold):
    model.eval()
    results = []

    with torch.no_grad():
        for ii, (protein_data, drug_data, label, drug_id, protein_id) in enumerate(test_loader):
            protein_data = protein_data.to(device)
            drug_data = drug_data.to(device)

            out = model(protein_data, drug_data)
            prediction_prob = torch.sigmoid(out).cpu().numpy()

            for i in range(len(drug_id)):
                binary_prediction = 1 if prediction_prob[i] > best_threshold else 0

                result = {
                    'drug_id': drug_id[i],
                    'protein_id': protein_id[i],
                    'prediction_prob': float(prediction_prob[i]),
                    'binary_prediction': binary_prediction,
                    'true_label': label[i].item() if label[i] != -1 else 'N/A'
                }
                results.append(result)

            if (ii + 1) % 1 == 0:  # 每个batch都打印
                print(f"Processed {ii + 1}/{len(test_loader)} batches...")

    return results


def save_results(results, output_folder):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_folder, f"prediction_results_{timestamp}.csv")

    # 只保留需要的列
    df = pd.DataFrame(results)[['drug_id', 'protein_id', 'prediction_prob', 'binary_prediction', 'true_label']]
    df.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")
    print(f"Total predictions: {len(results)}")

    return output_file


def main(opt, device):
    setup_logging(opt.log_folder)

    test_dataset = ProteinDrugDataset(
        protein_folder=opt.protein_folder,
        drug_folder=opt.drug_folder,
        tsv_file=opt.tsv_file
    )

    print(f"Prediction dataset length: {len(test_dataset)}")

    # 使用与训练时相同的DataLoader设置
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    model_state_dict, best_threshold = load_model(opt.model_path)
    model = DTISite(params)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    print(f"Model loaded successfully from {opt.model_path}")
    print(f"Using threshold: {best_threshold}")

    print("Starting prediction...")
    prediction_results = predict(model, device, test_loader, best_threshold)

    output_file = save_results(prediction_results, opt.log_folder)

    print(f"\nPrediction completed! Results saved to: {output_file}")

    # 打印前几个结果作为示例
    print("\nFirst few predictions:")
    for i, result in enumerate(prediction_results[:5]):
        print(f"{i + 1}. Drug: {result['drug_id']}, Protein: {result['protein_id']}, "
              f"Prob: {result['prediction_prob']:.4f}, Binary: {result['binary_prediction']}, "
              f"Label: {result['true_label']}")


if __name__ == "__main__":
    class Options:
        def __init__(self):
            self.protein_folder = "/sample_test/occurrence/protein_graph"
            self.drug_folder = "/sample_test/occurrence/drug_graph"
            self.model_path = "/models/occurrence_model.pth"
            self.tsv_file = "/sample_test/occurrence/sample_data.tsv"
            self.log_folder = "/logs/occurrence"
            self.batch_size = 1


    opt = Options()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(opt, device)