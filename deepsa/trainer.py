import time
import torch
import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from .utils import gen_canonical_smiles

def return_pred_label(prob, threshold):
    """
    Return prediction label based on threshold
    
    Parameters:
        prob: Prediction probability
        threshold: Threshold value
        
    Returns:
        1 (positive class) or 0 (negative class)
    """
    if prob >= threshold:
        return 1
    else:
        return 0

def return_gt_label(gt_label, postive_label):
    """
    Return ground truth label
    
    Parameters:
        gt_label: Ground truth label
        postive_label: Positive class label
        
    Returns:
        1 (positive class) or 0 (negative class)
    """
    if gt_label == postive_label:
        return 1
    else:
        return 0

def test_on_extrnal(test_set, label, score_column, preidcted_array):
    """
    Evaluate model on external test set
    
    Parameters:
        test_set: Test set DataFrame
        label: Label column name
        score_column: Score column name
        preidcted_array: Prediction results
        
    Returns:
        Dictionary of evaluation metrics
    """
    test_set = test_set.join(pd.DataFrame(preidcted_array), how='left')
    test_score = {}
    test_set['state'] = test_set[label].apply(lambda x: return_gt_label(x, postive_label=score_column))
    fpr, tpr, thresholds = roc_curve(test_set['state'], test_set[score_column], pos_label=1)
    test_score['roc_auc'] = auc(fpr, tpr)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_thershold = list(roc.iloc[(roc.tf-0).abs().argsort()[:1]]['threshold'])[0]
    test_set['pred'] = test_set.apply(lambda x: return_pred_label(x[score_column], roc_thershold), axis=1)
    test_score['acc'] = accuracy_score(test_set['state'], test_set['pred'])
    test_score['specificity'] = recall_score(test_set['state'], test_set['pred'], pos_label=0)
    test_score['precision'] = precision_score(test_set['state'], test_set['pred'])
    test_score['recall'] = recall_score(test_set['state'], test_set['pred'])
    test_score['sensitivity'] = test_score['recall']
    test_score['f1'] = f1_score(test_set['state'], test_set['pred'])
    test_score['thershold'] = roc_thershold
    return test_score

def get_pretrained_model_config(pretrained_type):
    """
    Get pretrained model configuration
    
    Parameters:
        pretrained_type: Pretrained model type
        
    Returns:
        Dictionary of pretrained model configuration
    """
    pretrained_models = {
        "scibert": {
            "path": "allenai/scibert_scivocab_uncased",
            "batch_size_per_gpu": 8
        },
        "bert": {
            "path": "bert-base-uncased",
            "batch_size_per_gpu": 16
        },
        "roberta": {
            "path": "roberta-base",
            "batch_size_per_gpu": 8
        }
    }
    
    return pretrained_models.get(pretrained_type.lower(), pretrained_models["scibert"])

def training(train_set, model_name, pretrained_model, label):
    """
    Train DeepSA model
    
    Parameters:
        train_set: Training set DataFrame
        model_name: Model save path
        pretrained_model: Pretrained model configuration
        label: Label column name
        
    Returns:
        Trained model
    """
    common_hyperparameters = {
        "model.hf_text.checkpoint_name": pretrained_model['path'],
        "model.hf_text.max_text_len": 512,
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
    
    deppshape_hyperparameters = {**common_hyperparameters, **batch_hyperparameters}
    
    predictor_proseq = MultiModalPredictor(label=label, path=model_name, eval_metric='accuracy').fit(
        train_set, 
        presets=None,
        hyperparameters=deppshape_hyperparameters,
        time_limit=36000
    )
    
    return predictor_proseq

def train_model(train_data, model_path, pretrained_type="scibert", test_data=None, test_list_path=None):
    """
    Train DeepSA model
    
    Parameters:
        train_data: Training set DataFrame
        model_path: Model save path
        pretrained_type: Pretrained model type
        test_data: Test set DataFrame
        test_list_path: Test set list file path
        
    Returns:
        Trained model and evaluation results
    """
    # Check required columns
    if "smiles" not in train_data.columns:
        raise ValueError("Training data must contain 'smiles' column")
    
    if "label" not in train_data.columns:
        raise ValueError("Training data must contain 'label' column")
    
    # Standardize SMILES
    train_data["smiles"] = train_data["smiles"].apply(lambda x: gen_canonical_smiles(x))
    
    # Remove invalid SMILES
    train_data = train_data[train_data["smiles"] != "gen_smiles_faild"]
    
    # Get pretrained model configuration
    pretrained_model = get_pretrained_model_config(pretrained_type)
    
    # Train model
    print(f"Starting model training, using pretrained model: {pretrained_model['path']}")
    start_time = time.time()
    model = training(train_data, model_path, pretrained_model, "label")
    end_time = time.time()
    print(f"Model training completed, time used: {(end_time - start_time) / 60:.2f} minutes")
    
    # Evaluate model
    results = {}
    
    # If test set is provided
    if test_data is not None:
        if "smiles" not in test_data.columns or "label" not in test_data.columns:
            raise ValueError("Test data must contain 'smiles' and 'label' columns")
        
        # Standardize SMILES
        test_data["smiles"] = test_data["smiles"].apply(lambda x: gen_canonical_smiles(x))
        
        # Remove invalid SMILES
        test_data = test_data[test_data["smiles"] != "gen_smiles_faild"]
        
        # Predict
        print("Evaluating model on test set...")
        test_pred = model.predict_proba(test_data, as_pandas=True)
        
        # Calculate evaluation metrics
        test_score = test_on_extrnal(test_data, "label", "easy", test_pred)
        results["test_score"] = test_score
        print(f"Test set evaluation results: AUC={test_score['roc_auc']:.4f}, ACC={test_score['acc']:.4f}, F1={test_score['f1']:.4f}")
    
    # If test set list is provided
    if test_list_path is not None and os.path.exists(test_list_path):
        with open(test_list_path, "r") as f:
            test_list = [line.strip() for line in f.readlines()]
        
        for test_file in test_list:
            if not os.path.exists(test_file):
                print(f"Warning: Test file {test_file} does not exist, skipping")
                continue
            
            try:
                test_data = pd.read_csv(test_file)
                
                if "smiles" not in test_data.columns or "label" not in test_data.columns:
                    print(f"Warning: Test file {test_file} must contain 'smiles' and 'label' columns, skipping")
                    continue
                
                # Standardize SMILES
                test_data["smiles"] = test_data["smiles"].apply(lambda x: gen_canonical_smiles(x))
                
                # Remove invalid SMILES
                test_data = test_data[test_data["smiles"] != "gen_smiles_faild"]
                
                # Predict
                print(f"Evaluating model on test set {test_file}...")
                test_pred = model.predict_proba(test_data, as_pandas=True)
                
                # Calculate evaluation metrics
                test_score = test_on_extrnal(test_data, "label", "easy", test_pred)
                results[test_file] = test_score
                print(f"Test set {test_file} evaluation results: AUC={test_score['roc_auc']:.4f}, ACC={test_score['acc']:.4f}, F1={test_score['f1']:.4f}")
            except Exception as e:
                print(f"Error evaluating test set {test_file}: {e}")
    
    return model, results