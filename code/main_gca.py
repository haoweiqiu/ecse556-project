import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch import nn 
import copy

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc
from sklearn.model_selection import KFold
import os

torch.manual_seed(1)    # reproducible torch:2 np:3
np.random.seed(1)

from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().to(device), p.long().to(device), d_mask.long().to(device), p_mask.long().to(device))
        
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()            
        
        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)
        
        loss_accumulate += loss
        count += 1
        
        logits = logits.detach().cpu().numpy()
        
        label_ids = label.cpu().numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        
    loss = loss_accumulate / count
    
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr + 0.00001)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: "+ str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0,0] + cm1[1,1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0,0] / (cm1[0,0] + cm1[0,1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1,1] / (cm1[1,0] + cm1[1,1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss.item()


def main(fold_n, lr):
    config = BIN_config_DBPE()
    
    BATCH_SIZE = config['batch_size']
    train_epoch = 20
    
    loss_history = []
    
    model = BIN_Interaction_Flat(**config)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)
            
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    print('--- Data Preparation ---')
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6, 
              'drop_last': True}

    dataFolder = './dataset/BIOSNAP/full_data'
    df_train = pd.read_csv(os.path.join(dataFolder, 'train.csv'))
    df_val = pd.read_csv(os.path.join(dataFolder, 'val.csv'))
    df_test = pd.read_csv(os.path.join(dataFolder, 'test.csv'))
    
    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)
    
    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)
    
    max_auc = 0
    model_max = copy.deepcopy(model)
    
    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(train_epoch):
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(d.long().to(device), p.long().to(device), d_mask.long().to(device), p_mask.long().to(device))

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()
            
            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))
            
            loss = loss_fct(n, label)
            loss_history.append(loss.item())
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if i % 100 == 0:
                print(f'Epoch {epo + 1}, Iteration {i}, Loss: {loss.item()}')
            
        with torch.no_grad():
            auc, auprc, f1, logits, loss = test(validation_generator, model)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
            
            print(f'Validation Epoch {epo + 1}, AUROC: {auc}, AUPRC: {auprc}, F1: {f1}')
    
    print('--- Go for Testing ---')
    try:
        with torch.no_grad():
            auc, auprc, f1, logits, loss = test(testing_generator, model_max)
            print(f'Testing AUROC: {auc}, AUPRC: {auprc}, F1: {f1}, Loss: {loss}')
    except Exception as e:
        print('Testing failed:', e)

    return model_max, loss_history


if __name__ == "__main__":
    s = time()
    model_max, loss_history = main(1, 5e-6)
    e = time()
    print(f"Total Training Time: {e - s} seconds")

    # Save the loss history plot
    loss_history_filtered = [x for x in loss_history if x < 1]
    plt.plot(loss_history_filtered)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.savefig('loss_history_gca.png')  # Save plot as a PNG file
    print("Loss history plot saved as loss_history_gca.png")
