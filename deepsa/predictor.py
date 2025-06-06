import os
import torch
import pandas as pd
import numpy as np
import pkg_resources
import warnings

# Suppress FutureWarning about torch.load weights_only parameter
warnings.filterwarnings("ignore", category=FutureWarning, module="autogluon.multimodal.learners.base")

# Set torch float32 matmul precision to 'high' for better performance on Tensor Core GPUs
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision('high')
        print("Set torch float32 matmul precision to 'high' for better performance")
    except AttributeError:
        # Older versions of PyTorch don't have this function
        print("Using older PyTorch version that doesn't support setting float32 matmul precision")

# Handle NLTK data download errors
try:
    import nltk
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Try to download NLTK data, but don't fail if it doesn't work
    try:
        nltk.download('averaged_perceptron_tagger', quiet=True, raise_on_error=False)
        nltk.download('wordnet', quiet=True, raise_on_error=False)
        nltk.download('omw-1.4', quiet=True, raise_on_error=False)
    except Exception as e:
        print(f"Warning: Failed to download NLTK data: {e}")
        print("This is not critical and the model should still work.")
except ImportError:
    print("NLTK not found, skipping NLTK data download.")

from autogluon.multimodal import MultiModalPredictor
from rdkit import Chem
from .utils import smiles2HA, smiles2RS, smiles2RingNum, rule_of_five, gen_smiles

# Get model path
def get_model_path():
    # First check if there is an environment variable specifying the model path
    model_path = os.environ.get("DEEPSA_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        return model_path
    
    # Otherwise use the built-in model in the package
    try:
        model_path = pkg_resources.resource_filename("deepsa", "model")
        if os.path.exists(model_path):
            return model_path
    except:
        pass
    
    # If the model is not in the package, try to find it in the current directory
    if os.path.exists("DeepSA_model"):
        return "DeepSA_model"
    
    raise FileNotFoundError("DeepSA model not found. Please set DEEPSA_MODEL_PATH environment variable or place the model in the current directory.")

def make_prediction(dataset, model_path=None, standardized=False):
    """
    Use DeepSA model to predict the synthetic accessibility of compounds
    
    Parameters:
        dataset: DataFrame containing a 'smiles' column
        model_path: Model path, if None, use the default path
        standardized: Whether to standardize SMILES
        
    Returns:
        DataFrame containing prediction results
    """
    if model_path is None:
        model_path = get_model_path()
    
    print(f"Loading model from: {model_path}")
    predictor_deepsa = MultiModalPredictor.load(path=model_path, verbosity=0)
    
    if standardized:
        dataset['smiles'] = dataset['smiles'].apply(lambda x: gen_smiles(x, random=False, kekule=False))
    
    dataset.drop_duplicates(subset=['smiles'], keep='first', inplace=True)
    
    print("Running prediction...")
    sa_data = predictor_deepsa.predict_proba(dataset, as_pandas=True)
    
    # Print prediction result information for debugging
    print(f"Prediction result columns: {sa_data.columns.tolist()}")
    print(f"Prediction result shape: {sa_data.shape}")
    
    output_data = dataset.join(sa_data, how='left')
    
    # Add additional molecular descriptors
    output_data['HA_num'] = output_data.apply(lambda x: smiles2HA(x['smiles']), axis=1)
    output_data['RingSystem_num'] = output_data.apply(lambda x: smiles2RS(x['smiles']), axis=1)
    output_data['Ring_num'] = output_data.apply(lambda x: smiles2RingNum(x['smiles']), axis=1) 
    output_data['rule_of_five'] = output_data.apply(lambda x: rule_of_five(x['smiles']), axis=1)
    
    # Release resources
    del predictor_deepsa
    torch.cuda.empty_cache()
    
    return output_data

def predict_sa(smiles, model_path=None, standardized=False):
    """
    Predict the synthetic accessibility of a single SMILES
    
    Parameters:
        smiles: SMILES string
        model_path: Model path, if None, use the default path
        standardized: Whether to standardize SMILES
        
    Returns:
        Synthetic accessibility score (a float between 0-1, higher means easier to synthesize)
    """
    # Check if SMILES is valid
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Create DataFrame
    df = pd.DataFrame({"smiles": [smiles]})
    
    # Predict
    result = make_prediction(df, model_path, standardized)
    
    # Get the column names from the prediction result
    pred_columns = [col for col in result.columns if col not in ['smiles', 'HA_num', 'RingSystem_num', 'Ring_num', 'rule_of_five']]
    
    # If 'easy' column exists, use it; otherwise use the first prediction column
    sa_column = 'easy' if 'easy' in pred_columns else pred_columns[0]
    
    # Return synthetic accessibility score
    return {
        "SA_score": result.iloc[0][sa_column],  # Synthetic accessibility score
        "HA_num": result.iloc[0]["HA_num"],  # Heavy atom count
        "Ring_num": result.iloc[0]["Ring_num"],  # Ring count
        "RingSystem_num": result.iloc[0]["RingSystem_num"],  # Ring system count
        "rule_of_five": bool(result.iloc[0]["rule_of_five"]),  # Whether it complies with the rule of five
    }

def predict_sa_from_file(file_or_df, model_path=None, standardized=False, output_path=None):
    """
    Predict the synthetic accessibility of compounds from a CSV file or DataFrame
    
    Parameters:
        file_or_df: Input CSV file path or DataFrame, must contain a 'smiles' column
        model_path: Model path, if None, use the default path
        standardized: Whether to standardize SMILES
        output_path: Output CSV file path, if None, use input filename + _results.csv
        
    Returns:
        DataFrame containing prediction results
    """
    # Read data
    if isinstance(file_or_df, pd.DataFrame):
        data_csv = file_or_df
    else:
        try:
            data_csv = pd.read_csv(file_or_df)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
    
    # Check if it contains a 'smiles' column
    if "smiles" not in data_csv.columns:
        raise ValueError("Input CSV file must contain a 'smiles' column")
    
    # Predict
    result = make_prediction(data_csv, model_path, standardized)
    
    # Get the column names from the prediction result
    pred_columns = [col for col in result.columns if col not in ['smiles', 'HA_num', 'RingSystem_num', 'Ring_num', 'rule_of_five']]
    
    # If 'easy' column exists, use it; otherwise use the first prediction column
    sa_column = 'easy' if 'easy' in pred_columns else pred_columns[0]
    
    # Add SA_score column for consistency
    if sa_column != 'SA_score':
        result['SA_score'] = result[sa_column]
    
    # Save results
    if output_path is not None:
        result.to_csv(output_path, index=False)
    elif not isinstance(file_or_df, pd.DataFrame):
        output_basename = os.path.splitext(os.path.basename(file_or_df))[0]
        output_path = f"{output_basename}_results.csv"
        result.to_csv(output_path, index=False)
    
    return result