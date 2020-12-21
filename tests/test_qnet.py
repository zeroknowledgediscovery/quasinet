import unittest
from unittest import TestCase

import pandas as pd
import numpy as np

from quasinet.qnet import Qnet, qdistance, membership_degree, qdistance_matrix

class TestQnet( TestCase):
    DATA_DIR = 'examples/example_data/'

    def load_data(self):
        if not hasattr(self, 'X'):
            max_rows = 5
            X = pd.read_csv(self.DATA_DIR + 'cchfl_test.csv')
            self.features = list(X.columns.astype(str))[:max_rows]
            X = X.values.astype(str)[:, :max_rows]
            self.X = X

        return self.X

    def test_create_qnet(self):
        if not hasattr(self, 'myqnet'):
            X = self.load_data()
            myqnet = Qnet(feature_names=self.features, n_jobs=1)
            myqnet.fit(X)
            self.myqnet = myqnet

        return self.myqnet

    def test_qdistance(self):
        X = self.load_data()

        seq1 = X[1]
        seq2 = X[2]
        myqnet = self.test_create_qnet()
        qdist = qdistance(seq1, seq2, myqnet, myqnet) 
        self.assertGreaterEqual(qdist, 0)

    @staticmethod
    def compute_qdistance_matrix(seqs1, seqs2, myqnet1, myqnet2):
        """Compute the qdistance matrix without any optimizations."""
        qdist_matrix = np.empty((len(seqs1), len(seqs2)))
        for i, seq1 in enumerate(seqs1):
            for j, seq2 in enumerate(seqs2):
                qdist_matrix[i, j] = qdistance(seq1, seq2, myqnet1, myqnet2)

        return qdist_matrix

    def test_qdistance_matrix(self):
        X = self.load_data()
        qnet1 = self.test_create_qnet()

        qnet2 = self.test_create_qnet()

        seqs1 = X[:4]
        seqs2 = X[4:7]

        # test when the seqs are different and qnets are different
        expected_qdist_matrix = TestQnet.compute_qdistance_matrix(
            seqs1,
            seqs2,
            qnet1,
            qnet2)
        actual_qdist_matrix = qdistance_matrix(seqs1, seqs2, qnet1, qnet2)
        self.assertTrue(np.allclose(expected_qdist_matrix, actual_qdist_matrix))

        # test when the seqs are the same and qnet is the same
        expected_qdist_matrix = TestQnet.compute_qdistance_matrix(
            seqs1,
            seqs1,
            qnet1,
            qnet1)
        actual_qdist_matrix = qdistance_matrix(seqs1, seqs1, qnet1, qnet1)
        self.assertTrue(np.allclose(expected_qdist_matrix, actual_qdist_matrix))

    


    def test_membership_degree(self):
        X = self.load_data()
        seq1 = X[1]
        myqnet = self.test_create_qnet()
        mem_degree = membership_degree(seq1, myqnet)


if __name__ == "__main__":
    unittest.main()