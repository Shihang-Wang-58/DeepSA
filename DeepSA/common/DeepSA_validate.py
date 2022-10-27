import os
import sys
import time
import warnings
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def test(test_set, predictor_proseq):
    test_score = predictor_proseq.evaluate(test_set, metrics=['acc', 'f1', 'roc_auc', 'precision', 'recall'])
    return test_score

def validate_model(key, value):
    val_set = value[0]
    dataset_name = str(val_set.split("/")[-1].split(".")[0])
    model_path = value[1]
    model_name = model_path.split("/")[-1]
    global sim_label
    np.random.seed(1207)
    predictor_proseq = TextPredictor.load(path=model_path)
    data_set = pd.read_csv(val_set, dtype={'id':str, label:str, 'smiles':str})
    # data_set['smiles'] = data_set['smiles'].apply(lambda x:gen_canonical_smiles(x, kekule=True))
    val_data = data_set[['smiles',label]]
    test_score = test(val_data, predictor_proseq)
    global validation_results
    print(f"Test {model_path} on {val_set}: \n ACC: {str(test_score['acc'])}\n F1: {str(test_score['f1'])}\n AUROC: {str(test_score['roc_auc'])}\n Precision:{str(test_score['precision'])}\n Recall:{str(test_score['recall'])}")
    validation_results.loc[[key],["Model"]] = str(model_path)
    validation_results.loc[[key],["Validation_Set"]] = str(val_set)
    validation_results.loc[[key],["acc"]] = str(test_score['acc'])
    validation_results.loc[[key],["f1"]] = str(test_score['f1'])
    validation_results.loc[[key],["roc_auc"]] = str(test_score['roc_auc'])
    validation_results.loc[[key],["precision"]] = str(test_score['precision'])
    validation_results.loc[[key],["recall"]] = str(test_score['recall'])
    # re-pred whole dataset
    preidcted_array = predictor_proseq.predict_proba(val_data[['smiles']])
    output_data_set = data_set.join(pd.DataFrame(preidcted_array), how='left')
    output_data_set.to_csv(f"{model_name}_{dataset_name}_pred.csv", index=False, header=True, sep=',')
    del predictor_proseq
    torch.cuda.empty_cache()
    time.sleep(5)

if __name__ == '__main__':
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list())
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    model_name_path_list = sys.argv[1]
    model_list = []
    validation_csv_list = sys.argv[2]
    validation_list = []
    label = "label"
    if os.path.exists(model_name_path_list):
        with open(model_name_path_list) as f:
            for line in f.readlines():
                line=line.strip('\n')
                model_list.append(line)
            f.close()
        job_basename = model_name_path_list.split(".")[0]+validation_csv_list.split(".")[0]
    else:
        model_list = model_name_path_list.split(":")
        job_basename = sys.argv[3]
    with open(validation_csv_list) as f:
        for line in f.readlines():
            line=line.strip('\n')
            validation_list.append(line)
        f.close()
    job_dict = {}
    job_id = 1
    for model_path in model_list:
        for val_set in validation_list:
            job_dict[job_id] = [val_set, model_path]
            job_id += 1
    print(f"NOTE: {len(job_dict)} jobs in queue.")
    validation_results = pd.DataFrame(columns=["Model","Validation_Set",'acc', 'f1', 'roc_auc', 'precision', 'recall'], index=job_dict.keys())
    for key, value in job_dict.items():
        validate_model(key, value)
    validation_results.to_csv(str(job_basename+"_results.csv"), index=False, header=True, sep=',')

