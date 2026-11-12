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

    def test_ocsvm_uses_the_unsw_v1_configuration(self):
        self.assertEqual(
            ms.model_args('OCSVM', 'NF-UNSW-NB15'),
            ['--scaling-type', 'MaxAbsScaler', '--nu', '0.05', '--q', '0.99',
             '--learning-rate', '0.1'],
        )

    def test_ocsvm_uses_the_unsw_v2_configuration(self):
        self.assertEqual(
            ms.model_args('OCSVM', 'NF-UNSW-NB15-v2'),
            ['--scaling-type', 'MaxAbsScaler', '--nu', '0.05', '--q', '0.99',
             '--learning-rate', '0.3'],
        )

    def test_ocsvm_runs_the_shared_detector_script(self):
        self.assertEqual(ms.MODELS['OCSVM']['script'], 'ocsvm_cic.py')


if __name__ == '__main__':
    unittest.main()
