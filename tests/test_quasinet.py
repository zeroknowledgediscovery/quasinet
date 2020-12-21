import unittest

import pandas as pd

from quasinet.tree import Node, get_nodes
from quasinet.qnet import Qnet, qdistance, membership_degree

@unittest.skip('Correct')
class TestTree(unittest.TestCase):

    @staticmethod
    def initialize_node():
        root = Node(1) 
        root.left = Node(2) 
        root.right = Node(3) 
        root.left.left = Node(4) 
        root.left.right = Node(5) 

        return root

    def test_get_nodes(self):
        root = self.initialize_node()
        nodes = get_nodes(root)
        all_cols = set([node.col for node in nodes])
        self.assertEqual(
            all_cols,
            set(range(1, 6)))

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

    @unittest.skip('Correct')
    def test_qdistance(self):
        X = self.load_data()

        seq1 = X[1]
        seq2 = X[2]
        myqnet = self.test_create_qnet()
        qdist = qdistance(seq1, seq2, myqnet, myqnet) 

    @unittest.skip('Correct')
    def test_membership_degree(self):
        X = self.load_data()
        seq1 = X[1]
        myqnet = self.test_create_qnet()
        mem_degree = membership_degree(seq1, myqnet)

# unittest.main()     
if __name__ == "__main__":
    unittest.main()