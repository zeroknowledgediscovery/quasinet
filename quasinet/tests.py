import unittest

from .tree import Node, get_nodes

# @unittest.skip('Correct')
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

if __name__ == "__main__":
    unittest.main()