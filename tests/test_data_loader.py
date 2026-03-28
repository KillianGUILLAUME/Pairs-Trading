import unittest
import numpy as np
import torch
import sys
import os

# Ensure the parent directory is in the python path to import neural_nets
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_nets.data_loader import CryptoPairsDataset, kalman_filter_spread

class TestCryptoPairsDataset(unittest.TestCase):
    
    def test_kalman_filter_spread(self):
        np.random.seed(42)
        
        # Generate synthetic cointegrated series
        # log_B is a random walk
        log_b = np.cumsum(np.random.normal(0, 0.01, 1000))
        
        # spread is an AR(1) mean-reverting process
        spread = np.zeros(1000)
        for t in range(1, 1000):
            spread[t] = 0.9 * spread[t-1] + np.random.normal(0, 0.05)
            
        # log_A = spread + 2.5 * log_b + 1.2
        log_a = spread + 2.5 * log_b + 1.2
        
        price_a = np.exp(log_a)
        price_b = np.exp(log_b)
        
        burn_in = 300
        out_log_a, out_log_b, out_spread = kalman_filter_spread(price_a, price_b, burn_in=burn_in)
        
        self.assertEqual(len(out_log_a), 1000)
        self.assertEqual(len(out_log_b), 1000)
        self.assertEqual(len(out_spread), 1000)
        
        # Verify the out_spread captures the simulated spread
        # The correlation between out_spread[burn_in:] and spread[burn_in:] should be high
        corr = np.corrcoef(out_spread[burn_in:], spread[burn_in:])[0, 1]
        self.assertGreater(corr, 0.5, "The Kalman filter spread should positively correlate with the true synthetic spread.")
        
    def test_dataset_shapes_and_logic(self):
        np.random.seed(42)
        price_a = np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
        price_b = np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
        
        dataset = CryptoPairsDataset(price_a, price_b, seq_len=60, burn_in=100, compute_signature=False)
        self.assertEqual(len(dataset), 500 - 100 - 60)
        
        batch = dataset[0]
        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(batch.shape, (60, 3)) # seq_len=60, 3 features (logA, logB, spread)
        
        # Verify there are no NaNs
        self.assertFalse(torch.isnan(batch).any())
        
        # Check standardisation dimensions
        self.assertEqual(dataset.mean.shape[0], 3)
        self.assertEqual(dataset.std.shape[0], 3)
        
        # Check that the batch is somewhat standardized (not necessarily mean 0 or std 1 just for this batch, but standard sizes)
        # We just test that the features are correctly formed.

if __name__ == '__main__':
    unittest.main()
