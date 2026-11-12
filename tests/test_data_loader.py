import unittest
import numpy as np
import torch

from neural_nets.data_loader import CryptoPairsDataset, kalman_filter_spread


class TestCryptoPairsDataset(unittest.TestCase):

    def test_kalman_filter_spread(self):
        np.random.seed(42)
        log_b = np.cumsum(np.random.normal(0, 0.01, 1000))
        spread = np.zeros(1000)
        for t in range(1, 1000):
            spread[t] = 0.9 * spread[t - 1] + np.random.normal(0, 0.05)
        log_a = spread + 2.5 * log_b + 1.2
        burn_in = 300
        out_log_a, out_log_b, out_spread = kalman_filter_spread(np.exp(log_a), np.exp(log_b), burn_in=burn_in)
        self.assertEqual(len(out_log_a), 1000)
        self.assertEqual(len(out_spread), 1000)
        corr = np.corrcoef(out_spread[burn_in:], spread[burn_in:])[0, 1]
        self.assertGreater(corr, 0.5)

    def test_dataset_shapes_and_logic(self):
        np.random.seed(42)
        price_a = np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
        price_b = np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
        dataset = CryptoPairsDataset(price_a, price_b, seq_len=60, compute_signature=False)
        # (n - seq_len) fenêtres de départ, stride 1 ; deux features (log A, log B)
        self.assertEqual(len(dataset), 500 - 60)
        past, target = dataset[0]
        self.assertIsInstance(past, torch.Tensor)
        self.assertEqual(past.shape, (60, 2))
        self.assertEqual(target.shape, (2,))
        self.assertFalse(torch.isnan(past).any())
        self.assertEqual(dataset.mean.shape[0], 2)
        self.assertEqual(dataset.std.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
