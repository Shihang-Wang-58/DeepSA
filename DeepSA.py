import sys
import torch
import pandas as pd
import numpy as np
from autogluon.text import TextPredictor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def smiles2mw(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        MW = Descriptors.MolWt(mol)
    except:
        MW = 'smiles_unvaild'
    return MW

def smiles2HA(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        HA_num = mol.GetNumHeavyAtoms()
    except:
        HA_num = 'smiles_unvaild'
    return HA_num

def smiles2RingNum(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        Ring_num = mol.GetRingInfo().NumRings()
    except:
        Ring_num = 'smiles_unvaild'
    return Ring_num

def GetRingSystems(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems

def smiles2RS(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        RS_num = len(GetRingSystems(mol))
    except:
        RS_num = 'smiles_unvaild'
    return RS_num

def rule_of_five(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    nrb = Descriptors.NumRotatableBonds(mol)
    # psa = Descriptors.TPSA(mol)
    if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and logp >= -2 and nrb <= 10):
        return 1
    else:
        return 0

def gen_smiles(smiles, kekule=False, random=False):
    try:   
        mol = Chem.MolFromSmiles(smiles) 
        Chem.SanitizeMol(mol)
        random_smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule, doRandom=random)
    except:
        random_smiles = smiles
    return random_smiles

def make_prediction(dataset, model_path, standardized=False):
    predictor_deepsa = TextPredictor.load(path=model_path, verbosity=0)
    if standardized:
        dataset['smiles'] = dataset['smiles'].apply(lambda x:gen_smiles(x, random=False, kekule=False))
    dataset.drop_duplicates(subset=['smiles'], keep='first', inplace=True)
    sa_data = predictor_deepsa.predict_proba(dataset, as_pandas=True)
    output_data = dataset.join(sa_data, how='left')
    del predictor_deepsa
    torch.cuda.empty_cache()
    return output_data

if __name__ == '__main__':
    data_csv = pd.read_csv(sys.argv[1])
    model_path = sys.argv[2]
    data = make_prediction(data_csv, model_path)
    data['HA_num'] = data.apply(lambda x: smiles2HA(x['smiles']), axis=1)
    data['RingSystem_num'] = data.apply(lambda x: smiles2RS(x['smiles']), axis=1)
    data['Ring_num'] = data.apply(lambda x: smiles2RingNum(x['smiles']), axis=1) 
    data['rule_of_five'] = data.apply(lambda x: rule_of_five(x['smiles']), axis=1)
    output_basename = str(sys.argv[1].split("/")[-1].split(".")[0])
    data.to_csv(str(sys.argv[1].split("/")[-1].split(".")[0]+"_results.csv"), index=False, header=True, sep=',')
   


