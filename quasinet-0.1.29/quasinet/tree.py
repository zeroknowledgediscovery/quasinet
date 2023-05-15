class Node(object):
    """Decision node in tree

    Parameters
    ----------
    col : int
        Integer indexing the location of feature or column

    col_pval : float
        Probability value from permutation test for feature selection

    lthreshold : list
        List of items for taking the left edge down the tree

    rthreshold : list
        List of items for taking the right edge down the tree

    impurity : float
        Impurity measuring quality of split

    value : 1d array-like or float
        For classification trees, estimate of each class probability
        For regression trees, central tendency estimate

    left : Node
        Another Node

    right : Node
        Another Node

    label_frequency: dict
        Dictionary mapping label to its frequency
    """

    def __init__(self, 
                 col=None, 
                 col_pval=None, 
                 lthreshold=None,
                 rthreshold=None, 
                 impurity=None,
                 value=None, 
                 left=None, 
                 right=None, 
                 label_frequency=None):

        assert isinstance(left, Node) or left is None
        assert isinstance(right, Node) or right is None

        self.col = col
        self.col_pval = col_pval
        self.lthreshold = lthreshold
        self.rthreshold = rthreshold
        self.impurity = impurity
        self.value = value
        self.left = left
        self.right = right
        self.label_frequency = label_frequency

    def __repr__(self):
        return "Node(col={})".format(self.col)

    def __str__(self):
        return self.__repr__()


def get_nodes(root, get_leaves=True, get_non_leaves=True):
    """Traverse a tree and get all the nodes.

    TODO: may need to change this into an iterator for speed purposes

    If `get_leaves` and `get_non_leaves` are both `True`, then
    we will get all the nodes.

    Parameters
    ----------
    root : Node
        root node of the tree

    get_leaves : bool
        If true, we get leaf nodes

    get_non_leaves : bool
        If true, we get non leaf nodes.


    Returns
    -------
    output : list
        list of `Node`
    """

    current = root
    stack = []
    all_nodes = []
    
    if get_leaves and get_non_leaves:
        get_type = 'all'
    elif get_leaves and not get_non_leaves:
        get_type = 'leaf'
    elif not get_leaves and get_non_leaves:
        get_type = 'nonleaf'
    else:
        raise ValueError('We must get either leaves or non-leaves or both.')

    while True: 
        if current is not None: 
            stack.append(current) 
            current = current.left  

        elif(stack): 
            current = stack.pop() 

            if get_type == 'all':
                all_nodes.append(current)
            elif get_type == 'leaf':
                if current.left is None and current.right is None:
                    all_nodes.append(current)
            elif get_type == 'nonleaf':
                if current.left is not None or current.right is not None:
                    all_nodes.append(current)
            else:
                raise ValueError

            current = current.right  

        else: 
            break

    return all_nodes