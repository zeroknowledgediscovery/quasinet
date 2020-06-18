class Node(object):
    """Decision node in tree

    Parameters
    ----------
    col : int
        Integer indexing the location of feature or column

    col_pval : float
        Probability value from permutation test for feature selection

    threshold : float
        Best split found in feature

    impurity : float
        Impurity measuring quality of split

    value : 1d array-like or float
        For classification trees, estimate of each class probability
        For regression trees, central tendency estimate

    left_child : Node
        Another Node

    right_child : Node
        Another Node

    label_frequency: dict
        Dictionary mapping label to its frequency
    """

    def __init__(self, 
                 col=None, 
                 col_pval=None, 
                 threshold=None, 
                 impurity=None,
                 value=None, 
                 left_child=None, 
                 right_child=None, 
                 label_frequency=None):

        assert isinstance(left_child, Node) or left_child is None
        assert isinstance(right_child, Node) or right_child is None

        self.col         = col
        self.col_pval    = col_pval
        self.threshold   = threshold
        self.impurity    = impurity
        self.value       = value
        self.left_child  = left_child
        self.right_child = right_child
        self.label_frequency = label_frequency


def get_nodes(root):
    """Traverse a tree and get all the nodes.

    TODO: may need to change this into an iterator for speed purposes

    Parameters
    ----------
    root : Node
        root node of the tree


    Returns
    -------
    output : list
        list of `Node`
    """

    current = root
    stack = []
    all_nodes = []
    
    while True: 
        if current is not None: 
            
            stack.append(current) 
        
            current = current.left  

        elif(stack): 
            current = stack.pop() 
            all_nodes.append(current)
            current = current.right  

        else: 
            break

    return all_nodes