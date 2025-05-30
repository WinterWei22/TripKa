# -*- coding: utf-8 -*-
import lmdb
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import argparse
import os

def get_Confs(smiles, num_confs=5, max_attempts=1000, max_retries=1,seed=1):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)


    params = AllChem.ETKDG()
    params.maxAttempts = max_attempts
    params.useRandomCoords = True
    params.enforceChirality = False
    params.randomSeed = seed

    conformers = []
    retries = 0
   
    while not conformers and retries < max_retries:
        try:
            
            
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
            
           
            if conf_ids:
                conformers = []
                for conf_id in conf_ids:
                    conformer = mol.GetConformer(conf_id)
                    num_atoms = mol.GetNumAtoms()
                    
                    
                    coords = np.zeros((num_atoms, 3))
                    for atom_id in range(num_atoms):
                        pos = conformer.GetAtomPosition(atom_id)
                        coords[atom_id] = [pos.x, pos.y, pos.z]
                    
                    conformers.append(coords)
                print(f"successfully generated {len(conformers)} conformers")
            else:
                raise ValueError("conformer generation failed")
        except Exception as e:
            retries += 1
            print(f"re-trying ({retries}/{max_retries}):...")
            
    if not conformers:
        return None
    
    return conformers

def get_Confs_MM(smiles, num_confs=5, seed=0):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        conformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed)
        for conf_id in range(len(conformers)):
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            
        num_atoms = mol.GetNumAtoms()
        coordinates = np.zeros((num_confs, num_atoms, 3))

        for conf_id in range(num_confs):
            conf = mol.GetConformer(conf_id)
            for atom_id in range(num_atoms):
                pos = conf.GetAtomPosition(atom_id)
                coordinates[conf_id, atom_id] = [pos.x, pos.y, pos.z]
    except:
        return None
    
    return coordinates

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument('--i', type=int, default=0, help="ith conf")
    parser.add_argument('--name', type=str, default="sampl6", help="test set name")
    parser.add_argument('--num-conf', type=int, default=1, help="num of conf")
    parser.add_argument('--node', type=int, default=0, help="nodes")
    parser.add_argument('--nodes', type=int, default=0, help="nodes")
    parser.add_argument("--mm", default=False, action="store_true",help="generate conformers by MM optimization")
    args = parser.parse_args()
    # outfilename='debug/dwar_train.lmdb'
    
    name = args.name
    mid_name = 'MM' if args.mm else 'ETKDG'
    if name == 'sampl6':
        filename='./Datasets/pickle/sampl6.pickle'
        outfiledir = f'./tripka/examples/{mid_name}/conf{args.num_conf}/{args.i}/sampl6_macro_regen'
        outfilename = f'{outfiledir}/sampl6_macro_regen.lmdb'
    elif name == 'sampl7':
        filename='./Datasets/pickle/sampl7.pickle'
        outfiledir = f'./tripka/examples/{mid_name}/conf{args.num_conf}/{args.i}/sampl7_macro_regen'
        outfilename = f'{outfiledir}/sampl7_macro_regen.lmdb'
    elif name == 'sampl8':
        filename='./Datasets/pickle/sampl8.pickle'
        outfiledir = f'./tripka/examples/{mid_name}/conf{args.num_conf}/{args.i}/sampl8_macro_regen'
        outfilename = f'{outfiledir}/sampl8_macro_regen.lmdb'
    elif name == 'novartis_a':
        filename='./Datasets/pickle/novartis_acid.pickle'
        outfiledir = f'./tripka/examples/{mid_name}/conf{args.num_conf}/{args.i}/novartis_a'
        outfilename = f'{outfiledir}/novartis_a.lmdb'
    elif name == 'novartis_b':
        filename='./Datasets/pickle/novartis_base.pickle'
        outfiledir = f'./tripka/examples/{mid_name}/conf{args.num_conf}/{args.i}/novartis_b'
        outfilename = f'{outfiledir}/novartis_b.lmdb'
    elif name == 'dwar_small':
        filename='./Datasets/pickle/dwar-iBond.pickle'
        outfiledir = f'./tripka/examples/{mid_name}/conf{args.num_conf}/dwar_small'
        outfilename = f'{outfiledir}/dwar_small.lmdb'
    elif name == 'chembl_train':
        filename='./Datasets/pickle/chembl_train.pickle'
        outfiledir = f'./tripka/examples/{mid_name}/conf{args.num_conf}/chembl_small'
        outfilename = f'{outfiledir}/train.lmdb'
        if args.nodes != 0:
            outfilename = f'{outfiledir}/train_{args.node}.lmdb'
            print(f'saving path: {outfilename}')
    elif name == 'chembl_valid':
        filename='./Datasets/pickle/chembl_valid.pickle'
        outfiledir = f'./tripka/examples/{mid_name}/conf{args.num_conf}/chembl_small'
        outfilename = f'{outfiledir}/valid.lmdb'

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

    txn_write = env_new.begin(write = True)
    num_confs = args.num_conf
    smi_lens = []
    with open(filename, 'rb') as file:
        input = pickle.load(file)
        input = input.to_dict(orient='records')
        if args.nodes != 0:
            print(f'node {args.node} / total nodes {args.nodes}')
            node = int(args.node)
            input = [input[i::args.nodes] for i in range(args.nodes)][node-1]
        data_dicts = []
        wrong_mol = 0
        for row in tqdm(input):
            smiles = row['SMILES']
            left_smi = smiles.split('>>')[0].split(',')
            right_smi = smiles.split('>>')[1].split(',')

            smi_lens.append(len(left_smi[0]))
            conf_wrong = False
            for i, l in enumerate(row['FEATURES_A']):
                row['FEATURES_A'][i]['smi'] = left_smi[i]
                if args.mm:
                    confs = get_Confs_MM(left_smi[i], num_confs=num_confs, seed=args.i)
                else:
                    confs = get_Confs(left_smi[i], num_confs=num_confs, seed=args.i)
                    
                row['FEATURES_A'][i]['coordinates'] = confs if confs is not None and len(confs) == num_confs else np.tile(row['FEATURES_A'][i]['coordinates'], (num_confs, 1, 1))
                if num_confs == 1 : row['FEATURES_A'][i]['coordinates'] =  row['FEATURES_A'][i]['coordinates'][0]
                if confs is None or len(confs) != num_confs :  conf_wrong = True
            for i, r in enumerate(row['FEATURES_B']):
                row['FEATURES_B'][i]['smi'] = right_smi[i]   
                if args.mm:
                    confs = get_Confs_MM(right_smi[i], num_confs=num_confs, seed=args.i)
                else:
                    confs = get_Confs(right_smi[i], num_confs=num_confs, seed=args.i)
    
                row['FEATURES_B'][i]['coordinates'] = confs if confs is not None and len(confs) == num_confs else np.tile(row['FEATURES_B'][i]['coordinates'], (num_confs, 1, 1))
                if num_confs == 1 : row['FEATURES_B'][i]['coordinates'] =  row['FEATURES_B'][i]['coordinates'][0]
                if confs is None or len(confs) != num_confs :   conf_wrong = True
            if conf_wrong: wrong_mol += 1

            modified_row = {
            'ori_smi': row['SMILES'],
            'metadata_a': row['FEATURES_A'],
            'metadata_b': row['FEATURES_B'],
            'target': row['TARGET']
            }
            data_dicts.append(modified_row)
        
        print(f"Conf generate Wrong mol: {wrong_mol}")
        i = 0
        for line in data_dicts:
            if line:
                txn_write.put(f'{i}'.encode(), pickle.dumps(line))
                i += 1

    print('process {} lines'.format(i))
    txn_write.commit()
    env_new.close()
