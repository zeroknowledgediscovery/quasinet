import unittest

import pandas as pd

from quasinet.qnet import Qnet, qdistance, membership_degree

class TestQnet(unittest.TestCase):
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

    def test_membership_degree(self):
        X = self.load_data()
        seq1 = X[1]
        myqnet = self.test_create_qnet()
        mem_degree = membership_degree(seq1, myqnet)

 
if __name__ == "__main__":
    unittest.main()