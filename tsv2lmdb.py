import numpy as np
import lmdb
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import os
from tqdm import tqdm
import math

def get_Confs(mol, num_confs=5, max_attempts=1000, max_retries=1):
    # 嵌入参数设置
    params = AllChem.ETKDG()
    params.maxAttempts = max_attempts
    params.useRandomCoords = True  # 允许使用随机坐标
    params.enforceChirality = False  # 放松手性约束

    conformers = []
    retries = 0
    # 尝试生成构象，直到成功
    while not conformers and retries < max_retries:
        try:
            # 尝试嵌入构象
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
            
            # 检查是否生成了构象
            if conf_ids:
                conformers = []
                for conf_id in conf_ids:
                    conformer = mol.GetConformer(conf_id)
                    num_atoms = mol.GetNumAtoms()
                    
                    # 提取每个构象的坐标，并存为ndarray
                    coords = np.zeros((num_atoms, 3))
                    for atom_id in range(num_atoms):
                        pos = conformer.GetAtomPosition(atom_id)
                        coords[atom_id] = [pos.x, pos.y, pos.z]
                    
                    # 将坐标添加到列表
                    conformers.append(coords)
                # print(f"成功生成 {len(conformers)} 个构象")
            else:
                raise ValueError(f"未能生成构象: {mol}")
        except Exception as e:
            retries += 1
            print(f"生成构象失败，重新尝试({retries}/{max_retries}):...")
            
    if not conformers:
        return None
    
    return conformers

def smiles_to_atoms_coordinates_charge(smiles, num_confs=1, conf_type='ETKDG'):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无法从SMILES生成分子对象")
    
    mol = Chem.AddHs(mol)
    if conf_type == 'ETKDG':
        coordinates = get_Confs(mol, num_confs=num_confs)
    elif conf_type == 'MM':
        coordinates = get_Confs_MM(mol, num_confs=num_confs)

    AllChem.ComputeGasteigerCharges(mol)
    
    # 获取原子列表
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = [float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()]
    
    return atoms, coordinates, charges

def get_Confs_MM(mol, num_confs=5):
    try:
        conformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=AllChem.ETKDG())
        for conf_id in range(len(conformers)):
        # 对每个构象进行优化
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            
        num_atoms = mol.GetNumAtoms()
        coordinates = np.zeros((num_confs, num_atoms, 3))  # 初始化 ndarray

        for conf_id in range(num_confs):
            conf = mol.GetConformer(conf_id)
            for atom_id in range(num_atoms):
                pos = conf.GetAtomPosition(atom_id)
                coordinates[conf_id, atom_id] = [pos.x, pos.y, pos.z]
    except:
        return None
    
    return coordinates

def tsv2pickle(conf_type, dataset_name, total_samples):
    conf_type = conf_type
    dataset_name = dataset_name
    total_confs = total_samples
    data_path = f'preprocess/{dataset_name}_enumed.tsv'
    for i in range(0,total_confs):
        num_confs = 1
        # basic
        df_data = pd.read_csv(data_path, sep='\t')
        f_a = []
        f_b = []
        smiles = []
        target = []
        qm_features = []
        num_wrong = 0
        for index, row in df_data.iterrows():
            mol_left = row['SMILES'].split('>>')[0].split(',')
            mol_right = row['SMILES'].split('>>')[1].split(',')
            features_A = []
            features_B = [] 
            wrong_mol  = False 
            for mol in mol_left:
                atoms, coordinates, charges = smiles_to_atoms_coordinates_charge(mol, num_confs=num_confs,conf_type=conf_type)
                if coordinates is None or len(coordinates) != num_confs :
                    wrong_mol = True
                features_l = {'atoms':atoms, 'coordinates': coordinates, 'charges': charges}
                features_A.append(features_l)
            for mol in mol_right:
                atoms, coordinates, charges = smiles_to_atoms_coordinates_charge(mol, num_confs=num_confs,conf_type=conf_type)
                if coordinates is None or len(coordinates) != num_confs :
                    wrong_mol = True
                features_r = {'atoms':atoms, 'coordinates': coordinates, 'charges': charges}
                features_B.append(features_r)

            if wrong_mol == True:
                num_wrong+=1
                continue

            smiles.append(row['SMILES'])
            f_a.append(features_A)
            f_b.append(features_B)

            target.append(row['TARGET'])

        print(f'wrong mol: {num_wrong}')
        df = pd.DataFrame({
            'SMILES': smiles,
            'FEATURES_A': f_a,
            'FEATURES_B': f_b,
            'TARGET': target
        })
        pickle_dir = f'data/pickle/{dataset_name}/{conf_type}'
        pickle_file = f'{pickle_dir}/{i}.pickle'
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        if os.path.exists(pickle_file):
            os.remove(pickle_file)
        df.to_pickle(pickle_file)

def pickle2lmdb(conf_type, dataset_name, total_samples):

    # total_confs=10
    for i in range(total_samples):
        num_confs = 1
        filename=f'data/pickle/{dataset_name}/{conf_type}/{i}.pickle'
        outfiledir=f'tripka/examples/{conf_type}/{i}/{dataset_name}'
        outfilename=f'{outfiledir}/{dataset_name}.lmdb'

        if not os.path.exists(outfiledir):
            os.makedirs(outfiledir)
        if os.path.exists(outfilename):
            os.remove(outfilename)

        env_new = lmdb.open(
                outfilename,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(100e9),
            )

        def has_isolated_hydrogens(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False  # 如果 SMILES 无效

            # 遍历所有原子，检查是否有孤立的氢原子
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:  # 检查氢原子
                    neighbors = atom.GetNeighbors()
                    if len(neighbors) == 0:
                        return False  # 找到孤立的氢原子
            return True  # 没有孤立的氢原子

        txn_write = env_new.begin(write = True)

        smi_lens = []
        with open(filename, 'rb') as file:
            input = pickle.load(file)
            input = input.to_dict(orient='records')
            data_dicts = []
            index = 0
            wrong_num = 0 
            for row in tqdm(input):
                smiles = row['SMILES']
                left_smi = smiles.split('>>')[0].split(',')
                # print(left_smi)
                right_smi = smiles.split('>>')[1].split(',')
                # print(right_smi)
                if len(left_smi[0]) > 100:  # 筛选大于100的分子
                    continue
                smi_lens.append(len(left_smi[0]))

                wrong_mol = False
                for i, l in enumerate(row['FEATURES_A']):
                    if has_isolated_hydrogens(left_smi[i]) == False:   wrong_mol = True
                    row['FEATURES_A'][i]['smi'] = left_smi[i]
                    confs = row['FEATURES_A'][i]['coordinates']
                    if confs is None or len(confs) != num_confs :wrong_mol =True
                    # print(len(row['FEATURES_A'][i]['coordinates']))
                    # print(row['FEATURES_A'][i]['coordinates'][0].shape)
                    if num_confs == 1:
                        row['FEATURES_A'][i]['coordinates'] = row['FEATURES_A'][i]['coordinates'][0]
                    # confs = get_Confs(left_smi[i], num_confs=num_confs)
                    # confs = None
                    # row['FEATURES_A'][i]['coordinates'] = confs if confs is not None and len(confs) == num_confs else np.tile(row['FEATURES_A'][i]['coordinates'], (num_confs, 1, 1))

                for i, r in enumerate(row['FEATURES_B']):
                    if has_isolated_hydrogens(right_smi[i]) == False:   wrong_mol = True
                    row['FEATURES_B'][i]['smi'] = right_smi[i]  
                    confs = row['FEATURES_B'][i]['coordinates']
                    if confs is None or len(confs) != num_confs :wrong_mol =True
                    # print(len(row['FEATURES_A'][i]['coordinates']))
                    if num_confs == 1:
                        row['FEATURES_B'][i]['coordinates'] = row['FEATURES_B'][i]['coordinates'][0]
                    # confs = get_Confs(right_smi[i], num_confs=num_confs)
                    # confs = None
                    # row['FEATURES_B'][i]['coordinates'] = confs if confs is not None and len(confs) == num_confs else np.tile(row['FEATURES_B'][i]['coordinates'], (num_confs, 1, 1))
                
                if wrong_mol == True : 
                    wrong_num+=1
                    continue
                
                try:
                    row['TARGET'] = float(row['TARGET'])
                except:
                    print(f"{smiles}: {row['TARGET']}")
                    continue
                    
                assert isinstance(row['TARGET'], float), f"{row['TARGET']} is not a float"
                if math.isnan(row['TARGET']):
                    print(f"{smiles}: {row['TARGET']}")
                    continue
                modified_row = {
                'ori_smi': row['SMILES'],
                'metadata_a': row['FEATURES_A'],
                'metadata_b': row['FEATURES_B'],
                'target': float(row['TARGET']),
                }
                

                data_dicts.append(modified_row)
                index += 1

            print(f'wrong num : {wrong_num}')
            i = 0
            for line in data_dicts:
                if line:
                    txn_write.put(f'{i}'.encode(), pickle.dumps(line))
                    i += 1
            print(i)

        print('process {} lines'.format(index))
        txn_write.commit()
        env_new.close()