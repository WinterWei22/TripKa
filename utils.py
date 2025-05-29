import pickle
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import pickle
import torch
import numpy as np
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error

def cal_fold(path, path_2 = None):
    f = open(path, 'rb')
    res = pickle.load(f)

    pred = []
    label = []
    for r in res:
        pred.extend(r['predict'])
        label.extend(r['target'])
    pred = [tensor.cpu().numpy() for tensor in pred]
    label = [tensor.cpu().numpy() for tensor in label]
    
    if path_2 is not None:
        f = open(path_2, 'rb')
        res = pickle.load(f)
        pred_2 = []
        label_2 = []
        for r in res:
            pred_2.extend(r['predict'])
            label_2.extend(r['target'])
        pred_2 = [tensor.cpu().numpy() for tensor in pred_2]
        label_2 = [tensor.cpu().numpy() for tensor in label_2]
        pred.extend(pred_2)
        label.extend(label_2)

    
    print(len(label))
    mae = mean_absolute_error(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    # print(mae,', ', rmse)
    return mae, rmse

def load_result(path, path_2=None):
    f = open(path, 'rb')
    res = pickle.load(f)

    pred = []
    label = []
    for r in res:
        pred.extend(r['predict'])
        label.extend(r['target'])
    pred = [tensor.cpu().numpy() for tensor in pred]
    label = [tensor.cpu().numpy() for tensor in label]
    
    if path_2 is not None:
        f = open(path_2, 'rb')
        res = pickle.load(f)
        pred_2 = []
        label_2 = []
        for r in res:
            pred_2.extend(r['predict'])
            label_2.extend(r['target'])
        pred_2 = [tensor.cpu().numpy() for tensor in pred_2]
        label_2 = [tensor.cpu().numpy() for tensor in label_2]
        pred.extend(pred_2)
        label.extend(label_2)

    return pred, label

def cal_mean(data):
    means = [sum(row) / len(row) for row in data if len(row) > 0]
    return means

def polar_atom_num(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    polar_atom = 0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol != 'C': 
            polar_atom+=1

    return polar_atom

def get_smiles_polarnum(path):
    f = open(path, 'rb')
    res = pickle.load(f)
    smiles_list = []
    for r in res:
        smis = r['smi_name']
        smis = [smi.split('>>')[0].split(',')[0] for smi in smis]
        smiles_list.extend(smis)
    
    polar_num_list = [polar_atom_num(smi) for smi in smiles_list]

    return smiles_list, polar_num_list

def cal_matrix_confidence(path):
    f = open(path, 'rb')
    res = pickle.load(f)
    pred = []
    label = []
    confidence = []
    for r in res:
        pred.extend(r['predict'])
        label.extend(r['target'])
        confidence.extend(r['confidence_predict'])
    pred = [tensor.cpu().numpy() for tensor in pred]
    label = [tensor.cpu().numpy() for tensor in label]
    confidence = [torch.sigmoid(tensor).cpu().numpy() for tensor in confidence]
    # print(mae,', ', rmse)
    return pred, label, confidence

def remove_outliers(pred, label, upper, downer):
    """
    按样本（行）筛选离群点，同时同步调整预测值和标签
    :param pred: numpy array, 预测值，形状为 [2, 30]
    :param label: numpy array, 标签值，形状为 [2, 30]
    :return: 筛选后的预测值和标签，列表形式，每个样本的长度可能不同
    """
    cleaned_pred = []
    cleaned_label = []
    
    
    for pred_row, label_row in zip(pred, label):  # 遍历每一行
        # 计算离群点的上下边界（IQR 方法）
        q1 = np.percentile(pred_row, downer)  # 第一四分位数
        q3 = np.percentile(pred_row, upper)  # 第三四分位数
        iqr = q3 - q1
        lower_bound = q1 - 1 * iqr
        upper_bound = q3 + 1 * iqr
            
        # 筛选非离群点
        mask = (pred_row > lower_bound) & (pred_row < upper_bound)

        if not np.any(mask):
            # 计算预测值和标签的绝对误差
            median_idx = np.argmin(np.abs(pred_row - np.median(pred_row)))
            # 创建只保留中值点的mask
            mask = np.zeros_like(pred_row, dtype=bool)
            mask[median_idx] = True
        
        cleaned_pred.append(pred_row[mask])
        cleaned_label.append(label_row[mask])
    
    return cleaned_pred, cleaned_label

def analysis_res_conf_single(dataset, num_samples, conf_type):
    preds = []
    labels = []
    confidences = []
    polar_nums = []
    smiles_list=[]
    task=dataset
    confidence_thredhold = num_samples//2
    for i in range(0, num_samples):
        path = f'results/{task}/{conf_type}/{i}/tripka_confidence_{task}.out.pkl'
        if not os.path.exists(path):
            print(f'path:{path} not exists')
            continue
        pred, label, confidence = cal_matrix_confidence(path)
        smiles, polar_num = get_smiles_polarnum(path)
        preds.append(pred)
        labels.append(label)
        confidences.append(confidence)
        polar_nums.append(polar_num)
    smiles_list.extend(smiles)
    preds = np.array(preds).squeeze(axis=-1)
    labels = np.array(labels).squeeze(axis=-1)
    confidences = np.array(confidences).squeeze(-1)
    polar_nums = np.array(polar_nums)
    confidence_idx = np.argsort(confidences, axis=0)[:confidence_thredhold]
    confidence_preds = preds[confidence_idx, np.arange(preds.shape[1])]
    confidence_labels = labels[confidence_idx, np.arange(labels.shape[1])]
    confidence_polarnum = polar_nums[confidence_idx, np.arange(polar_nums.shape[1])]
    confidence_preds_mean = np.mean(confidence_preds, axis=0) if len(confidence_preds.shape) == 2 else confidence_preds
    confidence_labels_mean = np.mean(confidence_labels, axis=0) if len(confidence_labels.shape) == 2 else confidence_labels
    confidence_polarnum_mean = np.mean(confidence_polarnum, axis=0) if len(confidence_polarnum.shape) == 2 else confidence_polarnum

    pred_trans = np.transpose(confidence_preds,(1,0))
    label_trans = np.transpose(confidence_labels,(1,0))
    upper=60
    downer=25
    pred_filtered, label_filtered = remove_outliers(pred_trans, label_trans, upper, downer=downer)

    pred_mean_conf = np.mean(pred_filtered)
    label_mean_conf = np.mean(pred_filtered)

    return confidence_preds_mean, pred_mean_conf,label_mean_conf

def analysis_res_qm_single(dataset, num_samples, conf_type):
    """batch w/o outliers"""
    pred_all = []
    label_all = []
    task=dataset
    for i in range(0, num_samples):

        path_a = f'results/{task}/{conf_type}/{i}/tripka_qm_{task}.out.pkl'
        pred_a, label_a = load_result(path_a)
        pred_all.append(pred_a)
        label_all.append(label_a)

    pred_mean = np.mean(pred_all, axis=0)[0][0]
    label_mean = np.mean(label_all, axis=0)[0][0]

    return pred_mean, label_mean

def analysis_res_qm_batch(dataset, num_samples, conf_type):
    pred_all = []
    label_all = []
    smiles_list = []
    task=dataset
    for i in range(0, num_samples):

        path_a = f'results/{task}/{conf_type}/{i}/tripka_qm_{task}.out.pkl'
        pred_a, label_a = load_result(path_a)
        smiles, polar_num = get_smiles_polarnum(path_a)
        pred_all.append(pred_a)
        label_all.append(label_a)
    smiles_list.append(smiles)
    pred_trans = np.transpose(np.squeeze(np.array(pred_all)),(1,0))
    label_trans = np.transpose(np.squeeze(np.array(label_all)),(1,0))
    pred_mean = cal_mean(pred_trans)
    label_mean = cal_mean(label_trans)

    return pred_mean, label_mean, smiles_list[0]

def analysis_res_conf_batch(dataset, num_samples, conf_type):
    preds = []
    labels = []
    confidences = []
    polar_nums = []
    smiles_list=[]
    task=dataset
    for i in range(0, num_samples):
        path = f'results/{dataset}/{conf_type}/{i}/tripka_confidence_{dataset}.out.pkl'
        if not os.path.exists(path):
            continue
        pred, label, confidence = cal_matrix_confidence(path)
        smiles, polar_num = get_smiles_polarnum(path)
        preds.append(pred)
        labels.append(label)
        confidences.append(confidence)
        polar_nums.append(polar_num)
        
    smiles_list.extend(smiles)
    preds = np.array(preds).squeeze(axis=-1)
    labels = np.array(labels).squeeze(axis=-1)
    confidences = np.array(confidences).squeeze(-1)
    polar_nums = np.array(polar_nums)
    confidence_idx = np.argsort(confidences, axis=0)[:]
    confidence_preds = preds[confidence_idx, np.arange(preds.shape[1])]
    confidence_labels = labels[confidence_idx, np.arange(labels.shape[1])]
    confidence_polarnum = polar_nums[confidence_idx, np.arange(polar_nums.shape[1])]
    confidence_preds_mean = np.mean(confidence_preds, axis=0) if len(confidence_preds.shape) == 2 else confidence_preds
    confidence_labels_mean = np.mean(confidence_labels, axis=0) if len(confidence_labels.shape) == 2 else confidence_labels
    confidence_polarnum_mean = np.mean(confidence_polarnum, axis=0) if len(confidence_polarnum.shape) == 2 else confidence_polarnum
    pred_trans = np.transpose(np.squeeze(confidence_preds),(1,0))
    label_trans = np.transpose(np.squeeze(confidence_labels),(1,0))

    upper=60
    downer=25
    pred_filtered, label_filtered = remove_outliers(pred_trans, label_trans, upper, downer=downer)

    pred_mean_conf = cal_mean(pred_filtered)
    label_mean_conf = cal_mean(label_filtered)

    return confidence_preds_mean, pred_mean_conf, label_mean_conf, smiles_list[0]
