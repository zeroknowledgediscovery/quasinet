import numpy as np

from .tree import get_nodes
from .utils import scientific_notation

class GraphvizExporter(object):
    """Export the tree using graphviz.

    Parameters
    ----------
    outfile : Qnet
        A Qnet instance

    response_name : str
        Name of the y variable that we are predicting

    feature_names : list
        Names of each of the features

    text_color : str
        Color to set the text

    edge_color : str
        Color to set the edges

    pen_width : int
        Width of pen for drawing boundaries
 
    Returns
    -------
    None
    """

    def __init__(self, 
                 tree,
                 outfile,
                 response_name,
                 feature_names, 
                 text_color='black', 
                 edge_color='gray',
                 edge_label_color='deepskyblue4',
                 pen_width=2,
                 background_color='transparent',
                 rotate=False):
        
        self.tree = tree
        self.outfile = outfile
        self.response_name = response_name
        self.feature_names = feature_names
        self.text_color = text_color
        self.edge_color = edge_color
        self.edge_label_color = edge_label_color
        self.pen_width = pen_width
        self.background_color = background_color
        self.rotate = rotate

    def export(self):
        nodes = get_nodes(
            self.tree.root, 
            get_leaves=True, 
            get_non_leaves=True)

        for i, node in enumerate(nodes):
            node.unique_id = i

        with open(self.outfile, 'w+') as f:
            self.write_tree_attributes(f)
            self.write_node_attributes(f)
            self.write_edge_attributes(f)

            leaf_node_ids = []
            final_string = "{rank = same;" + ';'.join(leaf_node_ids) + "}{rank = same; LEGEND;1;}}"

            f.write(final_string)

        for i, node in enumerate(nodes):
            delattr(node, 'unique_id')

    def write_tree_attributes(self, f):
        f.write('graph Tree {\n')
        f.write('node [shape=box')

        style = ['filled', 'rounded']
        style = ', '.join(style)

        f.write(''', style="{}", color="{}",penwidth="{}", \
                fontname=helvetica] ;\n'''.format(style,
                                                  self.text_color,
                                                  self.pen_width))

        f.write('''graph [ranksep=equally, splines=Curved, \
                bgcolor={}, dpi=600] ;\n'''.format(self.background_color))

        f.write('edge [fontname=helvetica, color={}] ;\n'.format(self.edge_color))

        if self.rotate:
            f.write('rankdir=LR ;\n')

        self.write_legend(f)


    def write_legend(self, f):
        label = ''

        if self.response_name is not None:
            label += "Response: {}\n".format(self.response_name)

        label += "Classes: {}\n".format(' | '.join(self.tree.labels_))

        f.write('LEGEND [label="{}",shape=note,align=left,\
                style=filled,fillcolor="slategray",\
                fontcolor="white",fontsize=10];\n'.format(label))

    def write_node_attribute(self, f, node):

        node_id = node.unique_id

        node_labels = ''

        if (node.left is None) and (node.right is None):
            max_index = np.argmax(node.value)
            max_val = node.value[max_index]
            prediction = self.tree.labels_[max_index]
            node_labels += '{}\nProb: {}'.format(prediction, max_val)
            node_color = '#E5FFCC'
        else:
            if self.feature_names is not None:
                node_labels += self.feature_names[node.col]
            pval = scientific_notation(node.col_pval)
            node_labels += " pval: {}".format(pval)
            node_color = '#ffffff'

        f.write('{} [label="{}"'.format(node_id, node_labels))
        f.write(', fillcolor="{}",fontcolor="{}"] ;\n'.format(node_color, self.text_color))

    def write_node_attributes(self, f):
        nodes = get_nodes(
            self.tree.root, 
            get_leaves=True, 
            get_non_leaves=True)

        for node in nodes:
            self.write_node_attribute(f, node)

    def write_edge_attribute(self, f, node):
        string_format = '{} -- {} [label="{}",fontcolor={},penwidth={}] ;\n'

        parent_id = node.unique_id
        left_node = node.left
        right_node = node.right

        

        if left_node is not None:
            child_id = left_node.unique_id
            child_label = '\\n'.join(node.lthreshold)
            
            f.write(string_format.format(parent_id,
                                         child_id,
                                         child_label,
                                         self.edge_label_color,
                                         self.pen_width))
        if right_node is not None:
            child_id = right_node.unique_id
            child_label = '\\n'.join(node.rthreshold)

            f.write(string_format.format(parent_id,
                                         child_id,
                                         child_label,
                                         self.edge_label_color,
                                         self.pen_width))

    def write_edge_attributes(self, f):
        nodes = get_nodes(
            self.tree.root, 
            get_leaves=True, 
            get_non_leaves=True)

        for node in nodes:
            self.write_edge_attribute(f, node)



