import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import multiexecution_sad_loda as ms


class TestModelArgs(unittest.TestCase):
    def test_loda_uses_normalizer_for_v1(self):
        self.assertEqual(
            ms.model_args('LODA', 'NF-UNSW-NB15'),
            ['--scaling-type', 'Normalizer', '--n-bins', '5', '--n-random-cuts', '100'],
        )

    def test_loda_uses_standard_scaler_for_v2(self):
        self.assertEqual(
            ms.model_args('LODA', 'NF-UNSW-NB15-v2'),
            ['--scaling-type', 'StandardScaler', '--n-bins', '20', '--n-random-cuts', '50'],
        )

    def test_sad_keeps_its_args_for_every_dataset(self):
        for dataset in ms.DATASETS:
            self.assertEqual(ms.model_args('SAD', dataset), ms.MODELS['SAD']['args'])


if __name__ == '__main__':
    unittest.main()
