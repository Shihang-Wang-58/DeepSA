import sys
import time
import torch
import pandas as pd
import numpy as np
from autogluon.text import TextPredictor
from sklearn.utils import shuffle

def training(train_set, model_name, pretrained_model, label):
    common_hyperparameters = {
        "model.hf_text.checkpoint_name": pretrained_model['path'],
        'optimization.learning_rate': 1.0e-3,
        "optimization.weight_decay": 1.0e-3, 
        "optimization.lr_schedule": "cosine_decay",
        "optimization.lr_decay": 0.9,
        "optimization.top_k_average_method": "greedy_soup",
        "optimization.top_k": 3, 
        "optimization.warmup_steps": 0.2, 
        "env.num_gpus": 1,
        "optimization.val_check_interval": 0.1, # for non-expand dataset: 0.5; for expand dataset: 0.1.
        "env.batch_size": 16*pretrained_model["batch_size_per_gpu"],
        'env.per_gpu_batch_size': pretrained_model["batch_size_per_gpu"],
    }
    if pretrained_model["batch_size_per_gpu"] > 512:
        batch_hyperparameters = {
            'optimization.max_epochs': 30,
            "optimization.patience": 30,
            "env.eval_batch_size_ratio": 8,
            "env.num_workers_evaluation": 8,
        }
    elif 256 < pretrained_model["batch_size_per_gpu"] <= 512:
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
            "env.eval_batch_size_ratio": 6,
            "env.num_workers_evaluation": 6,
        }
    elif 16 < pretrained_model["batch_size_per_gpu"] <= 64:
        batch_hyperparameters = {
            'optimization.max_epochs': 15,
            "optimization.patience": 15,
            "env.eval_batch_size_ratio": 4,
            "env.num_workers_evaluation": 4,
        }
    elif 0 < pretrained_model["batch_size_per_gpu"] <= 16:
        batch_hyperparameters = {
            'optimization.max_epochs': 10,
            "optimization.patience": 12,
            "env.eval_batch_size_ratio": 2,
            "env.num_workers_evaluation": 2,
        }
    deppshape_hyperparameters = { **common_hyperparameters, **batch_hyperparameters }
    predictor_proseq = TextPredictor(label=label, path=model_name, eval_metric='roc_auc').fit(
        train_set, 
        presets = None, # best_quality, high_quality,medium_quality_faster_train
        num_cpus = 12, 
        num_gpus = 1,
        plot_results = True,
        hyperparameters = deppshape_hyperparameters, 
        seed = training_random_seed,
    )
    predictor_proseq = TextPredictor.load(path=model_name)
    return predictor_proseq

def test(test_set, predictor_proseq):
    test_score = predictor_proseq.evaluate(test_set, metrics=['acc', 'f1', 'roc_auc', 'precision', 'recall'])
    return test_score

def training_and_test(data_set, label, model_name, pretrain_path, test_results, dataset_random_seed):
    # data process
    print(f"NOTE: Training {model_name} on the {label} Dataset..")
    if ":" in data_set:
        train_data = shuffle(pd.read_csv(data_set.split(":")[0]), random_state=dataset_random_seed)
        test_data = shuffle(pd.read_csv(data_set.split(":")[1]), random_state=dataset_random_seed)
        train_data = train_data[['smiles',label]]
        test_data = test_data[['smiles',label]]
    else:
        data_set = pd.read_csv(data_set)
        data_set = data_set[['smiles',label]]
        postive_data = shuffle(data_set[data_set[label]=="hs"], random_state=dataset_random_seed)
        negtive_data = shuffle(data_set[data_set[label]=="es"], random_state=dataset_random_seed)
        train_data = pd.concat([postive_data[1-45001:], negtive_data[1-45001:]], ignore_index=True)
        test_data = pd.concat([postive_data[45001-50001:], negtive_data[45001-50001:]], ignore_index=True)
    print('Training Set: Number=', len(train_data))
    print('Test Set: Number=', len(test_data))
    # training 
    predictor_proseq = training(train_data, model_name, pretrain_path, label)
    # test best model
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
    pretrained_models = {
        "ChemMLM": {'path':"DeepChem/ChemBERTa-77M-MLM", "batch_size_per_gpu": 96},
        "ChemMTR": {'path':"DeepChem/ChemBERTa-77M-MTR", "batch_size_per_gpu": 128},
        "TinBert": {'path':"prajjwal1/bert-tiny", "batch_size_per_gpu": 256},
        "SmELECTRA": {'path':"google/electra-small-discriminator", "batch_size_per_gpu": 48},
        "SmDeBERTa": {'path':"microsoft/deberta-v3-small", "batch_size_per_gpu": 32}, 
        "MuDeBERTa": {'path':"microsoft/mdeberta-v3-base", "batch_size_per_gpu": 16},
        "MinBert": {'path':"prajjwal1/bert-mini", "batch_size_per_gpu": 256},
        "RoBERTa": {'path':"roberta-base", "batch_size_per_gpu": 32},
        "DeBERTa": {'path':"microsoft/deberta-v3-base", "batch_size_per_gpu": 32}, 
        "GraphCodeBert": {'path':"microsoft/graphcodebert-base", "batch_size_per_gpu": 48}, 
    }
    # read data and build dataset
    data_set = sys.argv[1]
    pretrain_name = str(sys.argv[2])
    label = "label"
    # building
    model_name = str("DeepSA_"+pretrain_name)
    pretrain_model = pretrained_models[pretrain_name]
    test_results = pd.DataFrame(columns=['acc', 'f1', 'roc_auc', 'precision', 'recall'], index=[model_name])
    # training and test
    test_results = training_and_test(data_set, label, model_name, pretrain_model, test_results, dataset_random_seed)
    # output
    test_results.to_csv(str(model_name+"_results.csv"), index=True, header=True, sep=',')
