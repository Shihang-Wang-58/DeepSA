import os
import sys
import argparse
import pandas as pd
from .predictor import predict_sa, predict_sa_from_file
from .trainer import train_model

def predict_cli():
    """
    Command line interface: Predict synthetic accessibility of compounds
    """
    parser = argparse.ArgumentParser(description="DeepSA: Predict synthetic accessibility of compounds")
    parser.add_argument("input", help="Input file path (CSV format, must contain smiles column) or a single SMILES string")
    parser.add_argument("--model", "-m", help="Model path, default uses built-in model")
    parser.add_argument("--output", "-o", help="Output file path, default is input_results.csv")
    parser.add_argument("--standardize", "-s", action="store_true", help="Whether to standardize SMILES")
    
    args = parser.parse_args()
    
    # Determine if input is a file or SMILES string
    if os.path.exists(args.input) and args.input.endswith(".csv"):
        # File input
        try:
            result = predict_sa_from_file(
                args.input, 
                model_path=args.model, 
                standardized=args.standardize,
                output_path=args.output
            )
            print(f"Prediction completed, results saved to {args.output if args.output else os.path.splitext(os.path.basename(args.input))[0] + '_results.csv'}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # SMILES string input
        try:
            result = predict_sa(args.input, model_path=args.model, standardized=args.standardize)
            print("Prediction results:")
            print(f"SMILES: {args.input}")
            print(f"Synthetic accessibility score: {result['SA_score']:.4f}")
            print(f"Heavy atom count: {result['HA_num']}")
            print(f"Ring count: {result['Ring_num']}")
            print(f"Ring system count: {result['RingSystem_num']}")
            print(f"Rule of five compliance: {result['rule_of_five']}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

def train_cli():
    """
    Command line interface: Train DeepSA model
    """
    parser = argparse.ArgumentParser(description="DeepSA: Train synthetic accessibility prediction model")
    parser.add_argument("input", help="Input file path (CSV format, must contain smiles and label columns) or train_set:test_set format")
    parser.add_argument("output", help="Model output path")
    parser.add_argument("--test_list", "-t", help="Test set list file path")
    parser.add_argument("--pretrained", "-p", default="scibert", choices=["scibert", "bert", "roberta"], help="Pretrained model type")
    
    args = parser.parse_args()
    
    try:
        # Check input format
        if ":" in args.input:
            # train_set:test_set format
            train_path, test_path = args.input.split(":")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            # Train model
            train_model(
                train_data=train_data,
                test_data=test_data,
                model_path=args.output,
                pretrained_type=args.pretrained,
                test_list_path=args.test_list
            )
        else:
            # Single file format
            data = pd.read_csv(args.input)
            
            # Train model
            train_model(
                train_data=data,
                model_path=args.output,
                pretrained_type=args.pretrained,
                test_list_path=args.test_list
            )
        
        print(f"Training completed, model saved to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    predict_cli()