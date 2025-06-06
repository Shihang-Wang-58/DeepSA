# Advanced usage examples for DeepSA

import pandas as pd
import numpy as np
from deepsa import predict_sa, predict_sa_from_file
from deepsa.trainer import train_model
from deepsa.utils import gen_canonical_smiles, rule_of_five

# Example 1: Batch prediction of multiple SMILES and result analysis
def batch_prediction_example():
    # Create some test SMILES
    complex_molecules = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12",  # Propranolol
        "COC1=CC2=C(C=C1OC)C(=O)C(CC2)(C)C",  # Vanillin
    ]
    
    # Create DataFrame
    df = pd.DataFrame({"smiles": complex_molecules})
    
    # Prediction
    print("Batch prediction of multiple SMILES...")
    results = predict_sa_from_file(df)
    
    # Analyze results
    print("\nPrediction result analysis:")
    print(f"Average synthetic accessibility score: {results['easy'].mean():.4f}")
    print(f"Highest synthetic accessibility score: {results['easy'].max():.4f} (molecule: {results.loc[results['easy'].idxmax(), 'smiles']})")
    print(f"Lowest synthetic accessibility score: {results['easy'].min():.4f} (molecule: {results.loc[results['easy'].idxmin(), 'smiles']})")
    print(f"Proportion of molecules complying with rule of five: {results['rule_of_five'].mean() * 100:.1f}%")
    
    # Sort by synthetic accessibility score
    print("\nSorted by synthetic accessibility score:")
    sorted_results = results.sort_values(by="easy", ascending=False)
    for i, row in sorted_results.iterrows():
        print(f"SMILES: {row['smiles']}")
        print(f"  Synthetic accessibility score: {row['easy']:.4f}")
        print(f"  Heavy atom count: {row['HA_num']}")
        print(f"  Ring count: {row['Ring_num']}")
        print(f"  Ring system count: {row['RingSystem_num']}")
        print(f"  Rule of five compliance: {bool(row['rule_of_five'])}")

# Example 2: Create custom training data and train a model
def custom_training_example():
    # Note: This is just an example, actual training requires more data
    # Create some example data
    easy_molecules = [
        "CCO",  # Ethanol
        "CCCC",  # Butane
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CCN",  # Ethylamine
    ]
    
    hard_molecules = [
        "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=CC=C3",  # Terphenyl
        "C1CC2C3C1C4C5C2C3C45",  # Pentagonal dodecahedrane
        "C12=C3C4=C5C6=C1C7=C8C9=C%10C%11=C(C%12=C%13C%14=C%15C%16=C%12C%17=C%18C%19=C%16C%15=C%19C%18=C%17C%14=C%13C%11=C(C%10=C9C8=C7C6=C52)C43)C%20=C%21C%22=C%23C%24=C%20C%25=C%26C%27=C%24C%23=C%27C%26=C%25C%22=C%21",  # Super large ring
        "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=C(C=C3)C4=CC=C(C=C4)C5=CC=CC=C5",  # Pentaphenyl
        "C1CC2C3CC1CC(C2)C3",  # Ferrocene skeleton
    ]
    
    # Create training data
    train_smiles = easy_molecules + hard_molecules
    train_labels = ["easy"] * len(easy_molecules) + ["hard"] * len(hard_molecules)
    
    train_data = pd.DataFrame({
        "smiles": train_smiles,
        "label": train_labels
    })
    
    # Standardize SMILES
    train_data["smiles"] = train_data["smiles"].apply(gen_canonical_smiles)
    
    print("\nCreating custom training data:")
    print(train_data)
    
    # Train model
    # Note: Actual training requires more data, this is just an example
    print("\nTraining custom model...")
    print("Note: This is just an example, actual training requires more data")
    print("When using in practice, please uncomment the code below")
    
    # model, results = train_model(
    #     train_data=train_data,
    #     model_path="custom_deepsa_model",
    #     pretrained_type="scibert"
    # )
    # 
    # print("\nModel training completed, you can use the following code for prediction:")
    # print("from deepsa import predict_sa")
    # print("result = predict_sa('CCO', model_path='custom_deepsa_model')")

if __name__ == "__main__":
    print("Advanced usage examples for DeepSA\n")
    batch_prediction_example()
    custom_training_example()