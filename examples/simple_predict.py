# Simple DeepSA usage examples

import pandas as pd
from deepsa import predict_sa, predict_sa_from_file

# Example 1: Predict a single SMILES
smiles = "CCO"  # Ethanol
result = predict_sa(smiles)
print(f"SMILES: {smiles}")
print(f"Synthetic accessibility score: {result['DeepSA_score']:.4f}")
print(f"Heavy atom count: {result['HA_num']}")
print(f"Ring count: {result['Ring_num']}")
print(f"Ring system count: {result['RingSystem_num']}")
print(f"Rule of five compliance: {result['rule_of_five']}")
print("\n")

# Example 2: Predict multiple SMILES
smiles_list = ["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"]
df = pd.DataFrame({"smiles": smiles_list})
results = predict_sa_from_file(df, output_path="results.csv")
print("Prediction results for multiple SMILES:")
print(results[["smiles", "easy", "hard", "HA_num", "Ring_num", "RingSystem_num", "rule_of_five"]])