import pandas as pd
import sys
import os
import numpy as np
import warnings
import torch
from autogluon.text import TextPredictor
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
    print(dataset.head())
    dataset_sani, dataset_keku, dataset_random, dataset_temp1, dataset_temp2 = dataset, dataset, dataset, dataset, dataset
    dataset_sani[column] = dataset[column].apply(lambda x:gen_canonical_smiles(x, kekule=False))
    print(dataset_sani.head())
    dataset_keku[column] = dataset[column].apply(lambda x:gen_canonical_smiles(x, kekule=True))
    print(dataset_keku.head())
    for _ in range(expand_ratio):
        dataset_temp1[column] = dataset[column].apply(lambda x:gen_random_smiles(x, kekule=False))
        dataset_temp2[column] = dataset[column].apply(lambda x:gen_random_smiles(x, kekule=True))
        dataset_random = pd.concat([dataset_random, dataset_temp1, dataset_temp2], ignore_index=True)
    dataset = pd.concat([dataset_random, dataset_sani, dataset_keku], ignore_index=True)
    dataset = dataset[dataset[column]!='gen_smiles_faild']
    dataset.drop_duplicates(subset=[column], keep='first', inplace=True)
    return dataset

def deepsa_predictor(dataset, dataset_name, smiles_column, model_path):
    predictor = TextPredictor.load(path=model_path)
    model_name = model_path.split("/")[-1]
    preidcted_array = predictor.predict_proba(dataset[smiles_column], as_pandas=True)
    dataset = dataset.join(pd.DataFrame(preidcted_array), how='left')
    dataset.to_csv(f"{model_name}_{dataset_name}_deepsa.csv", index=False, header=True, sep=',')
    return dataset

if __name__ == "__main__":
    model_path_list = [ 
        "Model/DeepSA_ChemMLM", 
		"Model/DeepSA_ChemMTR", 
		"DeepSA_ChemMLM", 
        "DeepSA_ChemMTR",
    ]
    training_random_seed = 1102
    np.random.seed(training_random_seed)
    warnings.filterwarnings('ignore')
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    smiles_column = 'smiles'
    dataset = pd.read_csv(sys.argv[1], dtype={'id':str, 'smiles':str})
    dataset_name = sys.argv[1].split("/")[-1].split(".")[0]
    dataset = data_expand(dataset, smiles_column, 1)
    for model_path in model_path_list:
        deepsa_predictor(dataset, dataset_name, smiles_column, model_path)

