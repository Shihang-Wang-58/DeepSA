import sys
import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from rdkit import Chem

def gen_canonical_smiles(smiles, kekule=False):
    try:   
        mol = Chem.MolFromSmiles(smiles) 
        Chem.SanitizeMol(mol)
        canonical_smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule)
    except:
        canonical_smiles = 'gen_smiles_faild'
    return canonical_smiles

def gen_random_smiles(smiles, kekule=False):
    try:   
        mol = Chem.MolFromSmiles(smiles) 
        Chem.SanitizeMol(mol)
        random_smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule, doRandom=True)
    except:
        random_smiles = 'gen_smiles_faild'
    return random_smiles

def data_expand(dataset, column, expand_ratio):
    dataset_random, dataset_sani = dataset, dataset
    dataset_sani[column] = dataset_sani[column].apply(lambda x:gen_canonical_smiles(x, kekule=False))
    dataset = pd.concat([dataset, dataset_random, dataset_sani], ignore_index=True)
    for _ in range(expand_ratio):
        dataset_random[column] = dataset_random[column].apply(lambda x:gen_random_smiles(x, kekule=False))
    dataset = pd.concat([dataset, dataset_random], ignore_index=True)
    dataset = dataset[dataset[column]!='gen_smiles_faild']
    dataset.drop_duplicates(subset=[column], keep='first', inplace=True)
    return dataset

if __name__ == '__main__':
    # defalut params
    dataset_random_seed = 508
    training_random_seed = 1102
    expand_ratio = 5
    label = "label"
    # read data and build dataset
    data_set = pd.read_csv(sys.argv[1])
    postive_data = shuffle(data_set[data_set[label]=="hs"], random_state=dataset_random_seed)
    negtive_data = shuffle(data_set[data_set[label]=="es"], random_state=dataset_random_seed)
    train_data = pd.concat([postive_data[1-45001:], negtive_data[1-45001:]], ignore_index=True)
    test_data = pd.concat([postive_data[45001-50001:], negtive_data[45001-50001:]], ignore_index=True)
    train_data = data_expand(train_data, 'smiles', expand_ratio)
    test_data = data_expand(test_data, 'smiles', expand_ratio)
    # output
    train_data.to_csv(str(sys.argv[1].split(".")[0]+"_expand_train.csv"), index=False, header=True, sep=',')
    test_data.to_csv(str(sys.argv[1].split(".")[0]+"_expand_test.csv"), index=False, header=True, sep=',')
