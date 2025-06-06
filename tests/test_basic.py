# Basic functionality test script

import unittest
import pandas as pd
from deepsa import predict_sa, predict_sa_from_file
from deepsa.utils import smiles2mw, smiles2HA, smiles2RingNum, smiles2RS, rule_of_five, gen_smiles

class TestDeepSA(unittest.TestCase):
    
    def test_utils(self):
        """Test utility functions"""
        # Test SMILES to molecular weight
        mw = smiles2mw("CCO")
        self.assertIsInstance(mw, float)
        self.assertGreater(mw, 0)
        
        # Test SMILES to heavy atom count
        ha = smiles2HA("CCO")
        self.assertEqual(ha, 3)
        
        # Test SMILES to ring count
        ring_num = smiles2RingNum("c1ccccc1")
        self.assertEqual(ring_num, 1)
        
        # Test SMILES to ring system count
        rs_num = smiles2RS("c1ccccc1")
        self.assertEqual(rs_num, 1)
        
        # Test rule of five
        ro5 = rule_of_five("CCO")
        self.assertEqual(ro5, 1)
        
        # Test SMILES standardization
        std_smiles = gen_smiles("CCO")
        self.assertEqual(std_smiles, "CCO")
    
    def test_predict_single(self):
        """Test single SMILES prediction"""
        try:
            result = predict_sa("CCO")
            self.assertIsInstance(result, dict)
            self.assertIn("DeepSA_score", result)
            self.assertGreaterEqual(result["DeepSA_score"], 0)
            self.assertLessEqual(result["DeepSA_score"], 1)
            print("Single SMILES prediction test passed")
        except Exception as e:
            self.fail(f"Single SMILES prediction failed: {e}")
    
    def test_predict_batch(self):
        """Test batch SMILES prediction"""
        try:
            # Create test data
            smiles_list = ["CCO", "c1ccccc1"]
            df = pd.DataFrame({"smiles": smiles_list})
            
            # Predict
            results = predict_sa_from_file(df)
            
            # Validate results
            self.assertEqual(len(results), 2)
            self.assertIn("easy", results.columns)
            self.assertIn("hard", results.columns)
            print("Batch SMILES prediction test passed")
        except Exception as e:
            self.fail(f"Batch SMILES prediction failed: {e}")

if __name__ == "__main__":
    unittest.main()