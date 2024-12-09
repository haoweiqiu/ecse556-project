import torch
from torch import nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from PyBioMed.PyProtein import CTD
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    auc,
)
from sklearn.preprocessing import StandardScaler

torch.manual_seed(1)
np.random.seed(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def compute_ecfp4(smiles, n_bits=2048):
    """Computes ECFP4 fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)  # Return zero vector for invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)


def compute_pubchem(smiles):
    """Computes PubChem molecular descriptors."""
    # Example: Replace with actual PubChem feature computation or descriptor library
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(30)
    # Dummy example: Random features
    return np.random.rand(30)


def compute_psc(sequence):
    """Computes Protein Sequence Composition (PSC) features."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_len = len(sequence)
    counts = [sequence.count(aa) for aa in amino_acids]
    return np.array(counts) / seq_len  # Normalize by sequence length


def compute_ctd(sequence):
    """Computes CTD (Composition-Transition-Distribution) descriptors."""
    return np.array(list(CTD.CalculateCTD(sequence).values()))


def encode_features(data, drug_feature_type="ECFP4", protein_feature_type="PSC"):
    """Generates feature vectors for drugs and proteins."""
    features = []
    labels = []

    for _, row in data.iterrows():
        drug, target, label = row["SMILES"], row["Target Sequence"], row["Label"]
        
        # Drug features
        if drug_feature_type == "ECFP4":
            drug_features = compute_ecfp4(drug)
        elif drug_feature_type == "PubChem":
            drug_features = compute_pubchem(drug)
        else:
            raise ValueError(f"Unknown drug feature type: {drug_feature_type}")
        
        # Protein features
        if protein_feature_type == "PSC":
            protein_features = compute_psc(target)
        elif protein_feature_type == "CTD":
            protein_features = compute_ctd(target)
        else:
            raise ValueError(f"Unknown protein feature type: {protein_feature_type}")
        
        # Concatenate features
        combined_features = np.concatenate([drug_features, protein_features])
        features.append(combined_features)
        labels.append(label)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return np.array(features), np.array(labels)


def test(features, labels, model):
    """Tests the model and evaluates performance."""
    model.eval()
    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(features).squeeze()
    
    y_pred = logits.cpu().numpy()
    y_label = labels.cpu().numpy()

    # Compute metrics
    fpr, tpr, _ = roc_curve(y_label, y_pred)
    auc_score = auc(fpr, tpr)
    auprc = average_precision_score(y_label, y_pred)
    y_pred_s = (y_pred >= 0.5).astype(int)
    cm1 = confusion_matrix(y_label, y_pred_s)
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / cm1.sum()
    recall = recall_score(y_label, y_pred_s)
    precision = precision_score(y_label, y_pred_s)
    sensitivity1 = recall  # Sensitivity is the same as recall for binary classification
    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    f1 = f1_score(y_label, y_pred_s)

    # Print Metrics
    print(f"AUROC: {auc_score}")
    print(f"AUPRC: {auprc}")
    print(f"Confusion Matrix:\n{cm1}")
    print(f"Accuracy: {accuracy1}")
    print(f"Recall (Sensitivity): {sensitivity1}")
    print(f"Specificity: {specificity1}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")

    return auc_score, auprc, f1


def main():
    """Main function to run the training and testing pipeline."""
    dataFolder = './dataset/BIOSNAP/full_data'
    df_train = pd.read_csv(f"{dataFolder}/train.csv")
    df_test = pd.read_csv(f"{dataFolder}/test.csv")

    drug_feature_types = ["ECFP4", "PubChem"]
    protein_feature_types = ["PSC", "CTD"]
    results = []

    # Iterate over all combinations of drug and protein features
    for drug_feature_type in drug_feature_types:
        for protein_feature_type in protein_feature_types:
            print(f"Evaluating {drug_feature_type} + {protein_feature_type}")

            # Encode features
            train_features, train_labels = encode_features(df_train, drug_feature_type, protein_feature_type)
            test_features, test_labels = encode_features(df_test, drug_feature_type, protein_feature_type)

            # Define model
            input_dim = train_features.shape[1]
            model = LogisticRegression(input_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_function = nn.BCELoss()

            # Training loop
            for epoch in range(10):
                model.train()
                features = torch.tensor(train_features, dtype=torch.float32).to(device)
                labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

                optimizer.zero_grad()
                logits = model(features).squeeze()
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

            # Testing
            print("--- Testing ---")
            auc_score, auprc, f1 = test(test_features, test_labels, model)
            results.append({
                "Drug Feature": drug_feature_type,
                "Protein Feature": protein_feature_type,
                "AUROC": auc_score,
                "AUPRC": auprc,
                "F1": f1
            })

    # Print the best combination
    best_result = max(results, key=lambda x: x["AUROC"])
    print(f"Best Combination: {best_result}")


if __name__ == "__main__":
    main()
