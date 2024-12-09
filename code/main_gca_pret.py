import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, auc
from time import time
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
import copy
import codecs
from subword_nmt.apply_bpe import BPE

torch.manual_seed(1)
np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize ProtBert and ChemBERTa
prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
prot_model = BertModel.from_pretrained("Rostlab/prot_bert", add_pooling_layer=False).to(device)
drug_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
drug_model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", add_pooling_layer=False).to(device)

# Automatically determine embedding sizes
example_drug = "CCO"  # Example SMILES
example_protein = "MKTIIALSYIFCLVFADYKDDD"  # Example protein sequence

vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

with torch.no_grad():
    drug_encoded = drug_tokenizer(example_drug, return_tensors="pt", padding=True, truncation=True)
    drug_embedding_size = drug_model(**{k: v.to(device) for k, v in drug_encoded.items()}).last_hidden_state.size(-1)

    protein_encoded = prot_tokenizer(" ".join(example_protein), return_tensors="pt", padding=True, truncation=True)
    protein_embedding_size = prot_model(**{k: v.to(device) for k, v in protein_encoded.items()}).last_hidden_state.size(-1)

print(f"Detected drug embedding size: {drug_embedding_size}")
print(f"Detected protein embedding size: {protein_embedding_size}")


def preprocess_protein(sequence):
    """Preprocess protein sequences by replacing rare amino acids."""
    return " ".join(list(sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")))


def pad_or_truncate(embeddings, max_len):
    """Ensure embeddings have the correct sequence length."""
    if embeddings.size(0) > max_len:
        print("truncate")
        embeddings = embeddings[:max_len, :]
    elif embeddings.size(0) < max_len:
        print("pad")
        padding = torch.zeros(max_len - embeddings.size(0), embeddings.size(1), device=embeddings.device)
        embeddings = torch.cat((embeddings, padding), dim=0)
    return embeddings


def drug2emb_encoder(x):
    max_d = 50
    #max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


def embed_batch(batch, max_drug_seq, max_protein_seq, emb_size, drug_projector, protein_projector):
    """Embed drugs and proteins for a batch."""
    from models import Embeddings, Encoder_MultipleLayers  # Import required helper classes

    # Load configuration values
    config = BIN_config_DBPE()
    
    batch_size = len(batch["SMILES"])
    d_encoded_layers = torch.zeros(batch_size, max_drug_seq, emb_size, device=device)
    protein_embeddings = torch.zeros(batch_size, max_protein_seq, emb_size, device=device)
    labels = torch.zeros(batch_size, dtype=torch.float, device=device)

    # Initialize Embedding and Encoder layers using config
    demb = Embeddings(
        vocab_size=config["input_dim_drug"], 
        hidden_size=config["emb_size"], 
        max_position_size=config["max_drug_seq"], 
        dropout_rate=config["dropout_rate"]
    ).to(device)

    d_encoder = Encoder_MultipleLayers(
        n_layer=2,
        hidden_size=config["emb_size"],
        intermediate_size=config["intermediate_size"],
        num_attention_heads=config["num_attention_heads"],
        attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
        hidden_dropout_prob=config["hidden_dropout_prob"]
    ).to(device)

    for i in range(batch_size):
        drug = batch["SMILES"][i]
        protein = batch["Target Sequence"][i]
        label = batch["Label"][i]

        # Drug processing
        drug_idx, drug_mask = drug2emb_encoder(drug)  # Tokenized indices and mask
        drug_idx = torch.tensor(drug_idx, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dimension
        drug_mask = torch.tensor(drug_mask, dtype=torch.float, device=device).unsqueeze(0)  # Add batch dimension

        print(f"Batch {i}: drug_idx shape: {drug_idx.shape}, drug_mask shape: {drug_mask.shape}")

        # Mask expansion
        ex_d_mask = drug_mask.unsqueeze(1).unsqueeze(2)  # Expand dimensions for attention mask
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        print(f"Batch {i}: Expanded ex_d_mask shape: {ex_d_mask.shape}")

        # Pass through Embedding and Encoder layers
        d_emb = demb(drug_idx)  # Embedding
        print(f"Batch {i}: d_emb shape after embedding: {d_emb.shape}")
        
        d_encoded = d_encoder(d_emb.float(), ex_d_mask.float())  # Encoding
        print(f"Batch {i}: d_encoded shape after encoding: {d_encoded.shape}")
        
        d_encoded_layers[i] = d_encoded.squeeze(0)  # Remove batch dimension

        # Protein processing
        protein = preprocess_protein(protein)
        protein_encoded = prot_tokenizer(protein, return_tensors="pt", padding=True, truncation=True, max_length=max_protein_seq).to(device)
        print(f"Batch {i}: protein_encoded input shape: {protein_encoded['input_ids'].shape}")
        
        with torch.no_grad():
            protein_emb = prot_model(**protein_encoded).last_hidden_state.squeeze(0)
        print(f"Batch {i}: protein_emb shape after ProtBert: {protein_emb.shape}")
        
        protein_emb = pad_or_truncate(protein_emb, max_protein_seq)
        print(f"Batch {i}: protein_emb shape after pad_or_truncate: {protein_emb.shape}")
        if protein_projector:
            protein_emb = protein_projector(protein_emb.squeeze(0))
        protein_embeddings[i] = protein_emb.squeeze(0)
        print(f"Batch {i}: protein_embeddings shape after padding/truncation: {protein_embeddings[i].shape}")

        labels[i] = label

    print(f"Final d_encoded_layers shape: {d_encoded_layers.shape}")
    print(f"Final protein_embeddings shape: {protein_embeddings.shape}")
    print(f"Final labels shape: {labels.shape}")
    
    return d_encoded_layers, protein_embeddings, labels


def test(data_loader, model, max_drug_seq, max_protein_seq, emb_size, drug_projector, protein_projector):
    """Test function with detailed output."""
    y_pred, y_label = [], []
    model.eval()
    loss_accumulate, count = 0.0, 0.0

    for batch in data_loader:
        drug_emb, protein_emb, label = embed_batch(
            batch, max_drug_seq, max_protein_seq, emb_size, drug_projector, protein_projector
        )
        score = model(drug_emb, protein_emb)
        logits = torch.sigmoid(score).squeeze()
        loss_fct = torch.nn.BCELoss()
        loss = loss_fct(logits, label)
        loss_accumulate += loss.item()
        count += 1
        y_pred.extend(logits.detach().cpu().numpy())
        y_label.extend(label.cpu().numpy())

    loss = loss_accumulate / count
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    auc_k = auc(fpr, tpr)
    auprc = average_precision_score(y_label, y_pred)

    precision = tpr / (tpr + fpr + 0.00001)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    optimal_threshold = thresholds[5:][np.argmax(f1[5:])]

    print("Optimal threshold: ", optimal_threshold)
    y_pred_s = [1 if i >= optimal_threshold else 0 for i in y_pred]
    cm = confusion_matrix(y_label, y_pred_s)

    print("AUROC:", auc_k)
    print("AUPRC:", auprc)
    print("Confusion Matrix:\n", cm)
    print("Recall:", recall_score(y_label, y_pred_s))
    print("Precision:", precision_score(y_label, y_pred_s))
    accuracy = np.trace(cm) / np.sum(cm)
    print("Accuracy:", accuracy)

    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    return auc_k, auprc, loss


def main(fold_n, lr):
    config = BIN_config_DBPE()
    model = BIN_Interaction_Flat(**config).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    BATCH_SIZE = config["batch_size"]
    EMB_SIZE = config["emb_size"]

    drug_projector = (
        torch.nn.Linear(drug_embedding_size, EMB_SIZE).to(device) if drug_embedding_size != EMB_SIZE else None
    )
    protein_projector = (
        torch.nn.Linear(protein_embedding_size, EMB_SIZE).to(device) if protein_embedding_size != EMB_SIZE else None
    )

    print('--- Data Preparation ---')

    data_folder = "./dataset/BIOSNAP/full_data"
    df_train = pd.read_csv(f"{data_folder}/train.csv")[["SMILES", "Target Sequence", "Label"]]
    df_val = pd.read_csv(f"{data_folder}/val.csv")[["SMILES", "Target Sequence", "Label"]]
    df_test = pd.read_csv(f"{data_folder}/test.csv")[["SMILES", "Target Sequence", "Label"]]

    train_loader = DataLoader(df_train.to_dict("records"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(df_val.to_dict("records"), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(df_test.to_dict("records"), batch_size=BATCH_SIZE, shuffle=False)

    best_model, max_auc = None, 0
    loss_history = []

    train_epoch = 10  # Fixed train_epoch

    print('--- Go for Training ---')
    for epoch in range(train_epoch):
        print(f"Epoch {epoch + 1}/{train_epoch}")
        model.train()
        i = 0
        for batch in train_loader:
            print(f"batch: {i}")
            drug_emb, protein_emb, label = embed_batch(
                batch, config["max_drug_seq"], config["max_protein_seq"], EMB_SIZE, drug_projector, protein_projector
            )
            print(f"drug_emb: {drug_emb}")
            optimizer.zero_grad()
            score = model(drug_emb, protein_emb)
            print(f"score: {score}")
            logits = torch.sigmoid(score).squeeze()
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(logits, label)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            print(f'Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item()}')
            i = i + 1
        i = 0

        with torch.no_grad():
            val_auc, val_auprc, val_loss = test(
                val_loader, model, config["max_drug_seq"], config["max_protein_seq"], EMB_SIZE, drug_projector, protein_projector
            )
            print(f"Validation AUROC: {val_auc}, AUPRC: {val_auprc}, Loss: {val_loss}")
            if val_auc > max_auc:
                best_model = copy.deepcopy(model)
                max_auc = val_auc

    print("--- Testing ---")
    with torch.no_grad():
        test_auc, test_auprc, test_loss = test(
            test_loader, best_model, config["max_drug_seq"], config["max_protein_seq"], EMB_SIZE, drug_projector, protein_projector
        )
        print(f"Test Results: AUROC: {test_auc}, AUPRC: {test_auprc}, Loss: {test_loss}")

    return best_model, loss_history


if __name__ == "__main__":
    s = time()
    model_max, loss_history = main(1, 5e-3)
    e = time()
    print(f"Total Training Time: {e - s} seconds")

    # Save the loss history plot
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.savefig('loss_history_gca_pret.png')
    print("Loss history plot saved as 'loss_history_gca_pret.png'")
