import sys
import time
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from autogluon.tabular import TabularPredictor
from sklearn.utils import shuffle

def gen_fingerprint(smiles, fingerprint_type="ECFP4"):
    if fingerprint_type == "MACCS":
        from rdkit.Chem import MACCSkeys
        fgp = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
    elif fingerprint_type == "RDK":
        fgp = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    elif fingerprint_type == "AtomPairs":
        from rdkit.Chem.AtomPairs import Pairs
        fgp = Pairs.GetAtomPairFingerprint(Chem.MolFromSmiles(smiles))
    elif fingerprint_type == "TopologicalTorsion":
        from rdkit.Chem.AtomPairs import Torsions
        fgp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(Chem.MolFromSmiles(smiles))
    elif fingerprint_type == "ECFP4":
        from rdkit.Chem import AllChem
        fgp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2)
    elif fingerprint_type == "FCFP4":
        from rdkit.Chem import AllChem
        fgp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, useFeatures=True) 
    else:
        raise ValueError(
                    f"Unsupported fingerprint type {fingerprint_type}")
    fgp_array = np.frombuffer(bytes(fgp.ToBitString(), 'utf-8'), 'u1') - ord('0')
    return  fgp_array.tolist()

def feature_importance_analysis(predictor_proseq, test_data, best_model_name):
    original_feat_importance = predictor_proseq.feature_importance(data=test_data, model=best_model_name, feature_stage='original')
    transformed_feat_importance = predictor_proseq.feature_importance(data=test_data, model=best_model_name, feature_stage='transformed')
    transformed_model_feat_importance = predictor_proseq.feature_importance(data=test_data, model=best_model_name, feature_stage='transformed_model')
    original_feat_importance.to_csv(str(model_name+"_original_feat.csv"), index=False, header=True, sep=',')
    transformed_feat_importance.to_csv(str(model_name+"_transformed_feat.csv"), index=False, header=True, sep=',')
    transformed_model_feat_importance.to_csv(str(model_name+"_transformed_model.csv"), index=False, header=True, sep=',')

def leaderboard(predictor_proseq, data):
    leaderboard_table = predictor_proseq.leaderboard(data=data, extra_info=True, extra_metrics=['acc', 'f1', 'roc_auc', 'precision', 'recall'])
    leaderboard_table.to_csv(str(model_name+"_leaderboard.csv"), index=False, header=True, sep=',')

def training(train_set, model_name, label):
    # predictor_proseq = TabularPredictor(label=label, path=model_name, eval_metric='roc_auc', problem_type='binary').fit(
    #     train_set, 
    #     presets = "good_quality", # best_quality, high_quality, good_quality, medium_quality, optimize_for_deployment
    # )
    predictor_proseq = TabularPredictor.load(path=model_name, verbosity=3)
    return predictor_proseq

# def test_specific_model(test_set, predictor_proseq, model_name_to_evaluate):
#     test_score = predictor_proseq.evaluate(test_set, model=model_name_to_evaluate, detailed_report=True)
#     # pred_proba = predictor_proseq.predict_proba(test_set, model=model_name_to_evaluate)
#     # test_score = predictor_proseq.evaluate_predictions(test_set[label], pred_proba, detailed_report=True)
#     return test_score

# def test(test_set, predictor_proseq):
#     for model_name_to_evaluate in predictor_proseq.get_model_names():
#         new_test_score = test_specific_model(test_set, predictor_proseq, model_name_to_evaluate)
#         if 'test_score' not in locals() or new_test_score['roc_auc'] > test_score['roc_auc']:
#             test_score = new_test_score
#             best_model_name = model_name_to_evaluate
#     return best_model_name, test_score

def test(test_set, predictor_proseq):
    test_score = predictor_proseq.evaluate(test_set, detailed_report=True)
    return test_score

def training_and_test(data_set, label, model_name, fingerprint_name, test_results, dataset_random_seed):
    # data process
    print(f"NOTE: Training {model_name} on the {label} Dataset..")
    # if ":" in data_set:
    #     train_data = shuffle(pd.read_csv(data_set.split(":")[0]), random_state=dataset_random_seed)
    #     test_data = shuffle(pd.read_csv(data_set.split(":")[1]), random_state=dataset_random_seed)
    #     train_data['fingerprint'] = train_data['smiles'].apply(lambda x:gen_fingerprint(x, fingerprint_type=fingerprint_name))
    #     test_data['fingerprint'] = test_data['smiles'].apply(lambda x:gen_fingerprint(x, fingerprint_type=fingerprint_name))
    #     train_data = train_data[['fingerprint',label]]
    #     test_data = test_data[['fingerprint',label]]
    #     train_data = pd.DataFrame(train_data[label]).join(pd.DataFrame(train_data['fingerprint'].to_list()), how='left')
    #     test_data = pd.DataFrame(test_data[label]).join(pd.DataFrame(test_data['fingerprint'].to_list()), how='left')
    # else:
    #     data_set = pd.read_csv(data_set)
    #     data_set['fingerprint'] = data_set['smiles'].apply(lambda x:gen_fingerprint(x, fingerprint_type=fingerprint_name))
    #     data_set = data_set[['fingerprint',label]]
    #     data_set = pd.DataFrame(data_set[label]).join(pd.DataFrame(data_set['fingerprint'].to_list()), how='left')
    #     postive_data = shuffle(data_set[data_set[label]=="hs"], random_state=dataset_random_seed)
    #     negtive_data = shuffle(data_set[data_set[label]=="es"], random_state=dataset_random_seed)
    #     train_data = pd.concat([postive_data[1-360001:], negtive_data[1-360001:]], ignore_index=True)
    #     test_data = pd.concat([postive_data[360001-400001:], negtive_data[360001-400001:]], ignore_index=True)
    # train_data.to_csv(f"{model_name}_training_data.csv", index=False, header=True, sep=',')
    # test_data.to_csv(f"{model_name}_test_data.csv", index=False, header=True, sep=',')
    train_data = pd.read_csv(f"{model_name}_training_data.csv")
    test_data = pd.read_csv(f"{model_name}_test_data.csv")
    print('Training Set: Number=', len(train_data))
    print('Test Set: Number=', len(test_data))
    # training 
    predictor_proseq = training(train_data, model_name, label)
    # test best model
    # best_model_name, test_score = test(test_data, predictor_proseq)
    # train_score = test_specific_model(train_data, predictor_proseq, best_model_name)
    test_score = test(test_data, predictor_proseq)
    train_score = test(train_data, predictor_proseq)
    print('======== Job Report ========')
    print('Model performance on the training and validation set: ', model_name)
    print('ACC = {:.4f}'.format(train_score['acc']))
    print('F1 = {:.4f}'.format(train_score['f1']))
    print('ROC_AUC = {:.4f}'.format(train_score['roc_auc']))
    print('Precision = {:.4f}'.format(train_score['precision']))
    print('Recall = {:.4f}'.format(train_score['recall']))
    print('Model performance on the testing set: ', model_name)
    print('ACC = {:.4f}'.format(test_score['acc']))
    print('F1 = {:.4f}'.format(test_score['f1']))
    print('ROC_AUC = {:.4f}'.format(test_score['roc_auc']))
    print('Precision = {:.4f}'.format(test_score['precision']))
    print('Recall = {:.4f}'.format(test_score['recall']))
    test_results.loc[[model_name],['acc']] = str("{:.4f}/{:.4f}".format(train_score['acc'], test_score['acc']))
    test_results.loc[[model_name],['f1']] = str("{:.4f}/{:.4f}".format(train_score['f1'], test_score['f1']))
    test_results.loc[[model_name],['roc_auc']] = str("{:.4f}/{:.4f}".format(train_score['roc_auc'], test_score['roc_auc']))
    test_results.loc[[model_name],['precision']] = str("{:.4f}/{:.4f}".format(train_score['precision'], test_score['precision']))
    test_results.loc[[model_name],['recall']] = str("{:.4f}/{:.4f}".format(train_score['recall'], test_score['recall']))
    # analysis feature_importance
    # print(f"NOTE: The final best model is {best_model_name}")
    # predictor_proseq.set_model_best(best_model_name)
    feature_importance_analysis(predictor_proseq, test_data, None)
    # training analysis
    leaderboard(predictor_proseq, test_data)
    del predictor_proseq
    torch.cuda.empty_cache()
    time.sleep(5)
    print(
        f"\nNOTE: The {model_name} training successfully completed and the acc on the test set reached {test_score['acc']}.\n"
    )
    return test_results

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    dataset_random_seed = 508
    training_random_seed = 1102
    train_data_expand_ratio = 5
    test_data_expand_ratio = 10
    retained_origin = "yes"
    same_shape_ratio = 0.01
    np.random.seed(training_random_seed)
    # read data and build dataset
    data_set = sys.argv[1]
    fingerprint_name = str(sys.argv[2])
    label = "label"
    # building
    model_name = str("DeepSA_"+fingerprint_name)
    test_results = pd.DataFrame(columns=['acc', 'f1', 'roc_auc', 'precision', 'recall'], index=[model_name])
    # training and test
    test_results = training_and_test(data_set, label, model_name, fingerprint_name, test_results, dataset_random_seed)
    # output
    test_results.to_csv(str(model_name+"_results.csv"), index=True, header=True, sep=',')
