"""
Batch pKa
Input: tsv file contains keys("SMILES", "TARGET")
"""
import pandas as pd
import subprocess
from tsv2lmdb import tsv2pickle, pickle2lmdb
import sys
from utils import analysis_res_conf_batch, analysis_res_qm_batch, analysis_res_conf_single, analysis_res_qm_single
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import argparse
from pprint import pprint
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def enumerate(dataset, iter_num, data_path=None, mode='A'):
    mid_name=''
    for e in range(1,iter_num+1):
        if e == 1:
            if data_path == None:
                input_path=f'preprocess/{dataset}_raw.tsv'
            else: 
                input_path=data_path
        else:
            input_path=f'preprocess/{dataset}{mid_name}.tsv'
        mid_name += '_acid' if mode == 'A' else '_base'
        output_path=f'preprocess/{dataset}{mid_name}.tsv'
        command = [
                "python",           
                "enumerator.py",          
                "reconstruct",
                "-i", input_path,
                "-o", output_path,
                "-m", mode
            ]
        mode = 'B' if mode == 'A' else 'A'
        try: 
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"enumerating Wrong at iter：{e}")
        
        input_path=f'preprocess/{dataset}{mid_name}.tsv'
        mid_name += '_base' if mode == 'B' else '_acid'
        output_path=f'preprocess/{dataset}{mid_name}.tsv'
        command = [
            "python",           
            "enumerator.py",          
            "reconstruct",
            "-i", input_path,
            "-o", output_path,
            "-m", mode
        ]
        mode = 'B' if mode == 'A' else 'A'
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"enumerating Wrong at iter：{e}")

    data = pd.read_csv(output_path, sep='\t')
    data = data.loc[:, ['TARGET', 'SMILES']]
    final_path=f'preprocess/{dataset}_enumed.tsv'
    data.to_csv(final_path, sep='\t')

    print(f"successfully enumerated")

def calculate_metrics(df, actual_col, predicted_col):
    # 提取两列数据
    y_true = df[actual_col].values
    y_pred = df[predicted_col].values
    
    # 计算 MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算 R²
    r2 = r2_score(y_true, y_pred)
    
    # 返回结果
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse input")

    parser.add_argument("--smiles", type=str, help="if smiles else batch", default='')
    parser.add_argument('--data-path', type=str, help='data path', default='')
    parser.add_argument('--dataset', type=str, help='set a name for your task', default='SP7')
    parser.add_argument('--iter-num', type=int, help='enumeration iteration nums', default=4)
    parser.add_argument('--sample-num', type=int, help='samples num', default=4)
    parser.add_argument('--cuda-idx', type=int, help='cuda device', default=0)
    parser.add_argument('--micro', action="store_true", default=False)
    parser.add_argument('--micro-a', type=str, help='micro state acid smiles', default='')
    parser.add_argument('--micro-b', type=str, help='micro state base smiles', default='')
    parser.add_argument('--mode', type=str, help='inference mode', default='A')
    

    args = parser.parse_args()

    smiles = args.smiles
    data_path = args.data_path
    iter_num = args.iter_num
    total_samples = args.sample_num
    dataset = args.dataset
    flag = len(smiles) != 0 or args.micro   # smiles flag
    cuda_device = args.cuda_idx
    dataset = dataset+f'_iternum{iter_num}_samplenum{total_samples}'

    if len(smiles) != 0:
        target=0.0
        df = pd.DataFrame(columns=['SMILES', 'TARGET'])
        new_row = pd.DataFrame({'SMILES': [smiles], 'TARGET': [target]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(f'preprocess/{dataset}_raw.tsv', sep='\t')
        ori_smi = smiles
        enumerate(dataset, iter_num, mode=args.mode)
    elif args.micro:
        target=0.0
        assert len(args.micro_a) !=0, 'micro state of acid mol is required'
        assert len(args.micro_b) !=0, 'micro state of base mol is required'
        smiles = f'{args.micro_a}>>{args.micro_b}'
        df = pd.DataFrame(columns=['SMILES', 'TARGET'])
        new_row = pd.DataFrame({'SMILES': [smiles], 'TARGET': [target]})
        df = pd.concat([df, new_row], ignore_index=True)
        ori_smi = smiles
        df.to_csv(f'preprocess/{dataset}_enumed.tsv', sep='\t')
    else:
        enumerate(dataset, iter_num, data_path, mode=args.mode)
        df = pd.read_csv(data_path, sep='\t')
        ori_smi = df['SMILES'].values

    # tripka conf infer
    conf_type='ETKDG'
    tsv2pickle(conf_type='ETKDG', dataset_name=dataset, total_samples=total_samples)
    pickle2lmdb(conf_type='ETKDG', dataset_name=dataset, total_samples=total_samples)

    command = ["/bin/bash", "scripts/infer_tripka_confidence.sh", dataset, 
               str(total_samples), str(conf_type), str(cuda_device), str(8)]
    try:
        subprocess.run(command, check=True)
        print("infered successfully ")
    except subprocess.CalledProcessError as e:
        raise Exception(f"inference exception：{e}")

    if flag or args.micro:
        c_pred, c_pred_wo_outliers, label_conf = analysis_res_conf_single(dataset, total_samples, conf_type)
    else:
        c_pred, c_pred_wo_outliers, label_conf, smiles = analysis_res_conf_batch(dataset, total_samples, conf_type)

    # tripka qm infer
    conf_type='MM'
    tsv2pickle(conf_type=conf_type, dataset_name=dataset, total_samples=total_samples)
    pickle2lmdb(conf_type=conf_type, dataset_name=dataset, total_samples=total_samples)
    command = ["/bin/bash", "scripts/infer_tripka_qm.sh", dataset, 
               str(total_samples), str(conf_type), str(cuda_device), str(8)]
    
    try:
        subprocess.run(command, check=True)
        print("TripKa-qm infered successfully ")
    except subprocess.CalledProcessError as e:
        raise Exception(f"inference exception：{e}") 
    if flag:
        qm_pred, label_mean = analysis_res_qm_single(dataset, total_samples, conf_type)
    else:
        qm_pred, label_mean, smiles = analysis_res_qm_batch(dataset, total_samples, conf_type)
    
    pprint(vars(args), indent=2)
    if flag:
        qm_pred, label_mean = analysis_res_qm_single(dataset, total_samples, conf_type)
        print(f'mol: {dataset}, smiles: {smiles}')
        print(f'TripKa-conf: {c_pred_wo_outliers}, TripKa-qm: {qm_pred}')
        pred = 0.5*c_pred_wo_outliers + 0.5*qm_pred
        print(f'macro pKa predicted by TripKa: {pred}')
    else:
        pred_ensemble = []
        for c,q in zip(qm_pred, c_pred_wo_outliers):
            pred_ensemble.append(0.5*c + 0.5*q)
        print(len(ori_smi))
        print(len(c_pred_wo_outliers))
        print(len(qm_pred))
        print(len(pred_ensemble))
        print(len(label_mean))
        print(len(smiles))
        df = pd.DataFrame({
                'smiles': smiles,
                'preds_confidence': c_pred_wo_outliers,
                'preds_qm': qm_pred,
                'preds_ensemble': pred_ensemble,
                'labels': label_mean,
                'ori_smi': ori_smi
            })
        
        metrics_ensemble = calculate_metrics(df, 'labels', 'preds_ensemble')
        metrics_conf = calculate_metrics(df, 'labels', 'preds_confidence')
        metrics_qm = calculate_metrics(df, 'labels', 'preds_qm')

        save_path = f'analysis/{dataset}.csv'
        df.to_csv(save_path)
        print(f'The model inference was executed successfully. Results are retained at {save_path}.')
        print(f"TripKa output: MAE-{metrics_ensemble['MAE']}, RMSE-{metrics_ensemble['RMSE']}, R2-{metrics_ensemble['R2']}")
        print(f"TripKa-conf output: MAE-{metrics_conf['MAE']}, RMSE-{metrics_conf['RMSE']}, R2-{metrics_conf['R2']}")
        print(f"TripKa-qm output: MAE-{metrics_qm['MAE']}, RMSE-{metrics_qm['RMSE']}, R2-{metrics_qm['R2']}")
