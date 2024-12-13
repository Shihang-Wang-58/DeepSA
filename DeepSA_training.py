import sys
import time
import torch
import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor
from sklearn.utils import shuffle
from rdkit import Chem
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

def gen_canonical_smiles(smiles, kekule=False):
    try:   
        mol = Chem.MolFromSmiles(smiles) 
        Chem.SanitizeMol(mol)
        canonical_smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule)
    except:
        canonical_smiles = 'gen_smiles_faild'
    return canonical_smiles  

def return_pred_label(prob, threshold):
    if prob >= threshold:
        return 1
    else:
        return 0

def return_gt_label(gt_label, postive_label):
    if gt_label == postive_label:
        return 1
    else:
        return 0

def test_on_extrnal(test_set, label, score_column, preidcted_array):
    test_set = test_set.join(pd.DataFrame(preidcted_array), how='left')
    test_score = {}
    test_set['state'] = test_set[label].apply(lambda x:return_gt_label(x, postive_label=score_column))
    fpr, tpr, thresholds = roc_curve(test_set['state'], test_set[score_column], pos_label=1)
    test_score['roc_auc'] = auc(fpr, tpr)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
    roc_thershold = list(roc.iloc[(roc.tf-0).abs().argsort()[:1]]['threshold'])[0]
    test_set['pred'] = test_set.apply(lambda x: return_pred_label(x[score_column], roc_thershold), axis=1)
    test_score['acc'] = accuracy_score(test_set['state'], test_set['pred'])
    test_score['specificity'] = recall_score(test_set['state'],test_set['pred'], pos_label=0)
    test_score['precision'] = precision_score(test_set['state'],test_set['pred'])
    test_score['recall'] = recall_score(test_set['state'],test_set['pred'])
    test_score['sensitivity'] = test_score['recall']
    test_score['f1'] = f1_score(test_set['state'],test_set['pred'])
    test_score['thershold'] = roc_thershold
    # test_score = predictor_proseq.evaluate(test_set, metrics=['acc', 'f1', 'roc_auc', 'precision', 'recall'])
    return test_score

def training(train_set, model_name, pretrained_model, label):
    common_hyperparameters = {
        "model.hf_text.checkpoint_name": pretrained_model['path'],
        "model.hf_text.max_text_len" : 512,
        'optimization.learning_rate': 1.0e-3,
        "optimization.weight_decay": 1.0e-3, 
        "optimization.lr_schedule": "cosine_decay",
        "optimization.lr_decay": 0.9,
        "optimization.top_k_average_method": "greedy_soup",
        "optimization.top_k": 3, 
        "optimization.warmup_steps": 0.2, 
        "env.num_gpus": 1,
        "optimization.val_check_interval": 0.2, 
        "env.batch_size": 8,
        'env.per_gpu_batch_size': pretrained_model["batch_size_per_gpu"],
    }
    if pretrained_model["batch_size_per_gpu"] > 256:
        batch_hyperparameters = {
            'optimization.max_epochs': 30,
            "optimization.patience": 20,
            "env.eval_batch_size_ratio": 6,
            "env.num_workers_evaluation": 6,
        }
    elif 64 < pretrained_model["batch_size_per_gpu"] <= 256:
        batch_hyperparameters = {
            'optimization.max_epochs': 20,
            "optimization.patience": 15,
            "env.eval_batch_size_ratio": 4,
            "env.num_workers_evaluation": 2,
        }
    elif 16 < pretrained_model["batch_size_per_gpu"] <= 64:
        batch_hyperparameters = {
            'optimization.max_epochs': 15,
            "optimization.patience": 15,
            "env.eval_batch_size_ratio": 4,
            "env.num_workers_evaluation": 2,
        }
    elif 0 < pretrained_model["batch_size_per_gpu"] <= 16:
        batch_hyperparameters = {
            'optimization.max_epochs': 10,
            "optimization.patience": 12,
            "env.eval_batch_size_ratio": 2,
            "env.num_workers_evaluation": 2,
        }
    deppshape_hyperparameters = { **common_hyperparameters, **batch_hyperparameters }
    predictor_proseq = MultiModalPredictor(label=label, path=model_name, eval_metric='accuracy').fit(
        train_set, 
        presets = None, # best_quality, high_quality,medium_quality_faster_train
        column_types = {"smiles": "text", label: "categorical"},
        hyperparameters = deppshape_hyperparameters, 
        seed = training_random_seed,
    )
    return predictor_proseq

def train_model(data_set, label, model_name, pretrain_path, dataset_random_seed):
    # data process
    print(f"NOTE: Training {model_name} on the {label} Dataset..")
    if ":" in data_set:
        train_data = shuffle(pd.read_csv(data_set.split(":")[0]), random_state=dataset_random_seed)
        test_data = shuffle(pd.read_csv(data_set.split(":")[1]), random_state=dataset_random_seed)
        postive_data = shuffle(train_data[train_data[label]=="hs"], random_state=dataset_random_seed)
        negtive_data = shuffle(train_data[train_data[label]=="es"], random_state=dataset_random_seed)
        sample_size = np.min([len(postive_data), len(negtive_data)])
        train_data = pd.concat([postive_data[1-sample_size:], negtive_data[1-sample_size:]], ignore_index=True)
        train_data = train_data[['smiles',label]]
        test_data = test_data[['smiles',label]]
    else:
        data_set = pd.read_csv(data_set)
        data_set = data_set[['smiles',label]]
        postive_data = shuffle(data_set[data_set[label]=="hs"], random_state=dataset_random_seed)
        negtive_data = shuffle(data_set[data_set[label]=="es"], random_state=dataset_random_seed)
        train_data = pd.concat([postive_data[1-360001:], negtive_data[1-360001:]], ignore_index=True)
        test_data = pd.concat([postive_data[360001-400001:], negtive_data[360001-400001:]], ignore_index=True)
    print('Training Set: Number=', len(train_data))
    print('Test Set: Number=', len(test_data))
    # training 
    predictor_proseq = training(train_data, model_name, pretrain_path, label)
    return predictor_proseq

def validate_model(validation_list, predictor_proseq):
    validation_results = pd.DataFrame(columns = ["Model", "Validation_Set", "prob_label", 'acc', 'f1', 'roc_auc', 'precision', 'recall', 'specificity', 'sensitivity', 'thershold'])
    for val_set in validation_list:
        dataset_name = str(val_set.split("/")[-1].split(".")[0])
        data_set = pd.read_csv(val_set, dtype={'id':str, label:str, 'smiles':str})
        # data_set['smiles'] = data_set['smiles'].apply(lambda x:gen_canonical_smiles(x, kekule=False))
        val_data = data_set[['smiles',label]]
        preidcted_array = predictor_proseq.predict_proba(val_data[['smiles']])
        output_data_set = data_set.join(pd.DataFrame(preidcted_array), how='left')
        output_data_set.to_csv(f"{model_name}/{model_name}_{dataset_name}_pred.csv", index=False, header=True, sep=',')
        for prob_label in ['hs', 'es']:
            test_score = test_on_extrnal(val_data, label, prob_label, preidcted_array)
            print(f"Test {prob_label} of {model_name} on {val_set}: \n Thershold: {str(test_score['thershold'])}\n ACC: {str(test_score['acc'])}\n F1: {str(test_score['f1'])}\n AUROC: {str(test_score['roc_auc'])}\n Precision:{str(test_score['precision'])}\n Recall:{str(test_score['recall'])}\n Specificity: {str(test_score['specificity'])}\n Sensitivity: {str(test_score['sensitivity'])}\n")
            validation_report = [ (model_name, str(val_set), prob_label, str(test_score['acc']), str(test_score['f1']), str(test_score['roc_auc']), str(test_score['precision']), str(test_score['recall']), str(test_score['specificity']), str(test_score['sensitivity']), str(test_score['thershold']))]
            new_row = pd.DataFrame(validation_report, columns = ["Model", "Validation_Set", "prob_label", 'acc', 'f1', 'roc_auc', 'precision', 'recall', 'specificity', 'sensitivity', 'thershold'])
            validation_results = pd.concat([validation_results, new_row], ignore_index=True)
    return validation_results

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    dataset_random_seed = 508
    training_random_seed = 3407
    np.random.seed(training_random_seed)
    pretrained_models = {
        "SmELECTRA": {'path':"google/electra-small-discriminator", "batch_size_per_gpu": 256},
    }
    # read data and build dataset
    data_set = sys.argv[1]
    model_basename = str(sys.argv[2])
    validation_csv_list = sys.argv[3]
    validation_list = []
    label = "label"
    with open(validation_csv_list) as f:
        for line in f.readlines():
            line=line.strip('\n')
            validation_list.append(line)
        f.close()
    label = "label"
    # building
    for pretrain_name,pretrain_model in pretrained_models.items():
        model_name = str(model_basename+"_"+pretrain_name)
        # training and test
        predictor_proseq = train_model(data_set, label, model_name, pretrain_model, dataset_random_seed)
        validation_results = validate_model(validation_list, predictor_proseq)
        validation_results.to_csv(str(model_name+"_validation_results.csv"), index=False, header=True, sep=',')   
        del predictor_proseq
        torch.cuda.empty_cache()
        time.sleep(5)



