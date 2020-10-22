import numpy as np

from .tree import get_nodes
from .utils import scientific_notation


def _rgb_to_hex(rgb):
    """Convert RGB to hex for colors.

    Parameters
    ----------
    rgb : 1-d like array
        RGB array of size 3.

    Returns
    -------
    hex_repr : str
        Hexidecimal representation of rgb.
    """

    hex_repr = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
    return hex_repr

def _color_brew(n, alpha=1.0):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    alpha : float
        Factor to upscale all the colors.

    Returns
    -------
    color_array : 2d array-like
        Arrat of size (n, 3), where each row is of form (R, G, B), which are 
        components of each color.
    """

    color_array = []

    # initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]

        # shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]

        # perform upscaling
        rgb = np.array(rgb) * alpha
        np.putmask(rgb, rgb > 255, 255)

        color_array.append(rgb.astype(int))

    return np.array(color_array)


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
 
    dpi : int
        Image resolution
        
    rotate : bool
        If True, rotate the tree

    add_legend : bool
        If True, add a legend to the tree

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
                 font_size=10,
                 edge_label_color='deepskyblue4',
                 pen_width=2,
                 background_color='transparent',
                 dpi=200,
                 rotate=False,
                 add_legend=True):
        
        self.tree = tree
        self._total_samples = sum(tree.root.label_frequency.values())
        self.outfile = outfile
        self.response_name = response_name
        self.feature_names = feature_names
        self.text_color = text_color
        self.edge_color = edge_color
        self.font_size = font_size
        self.edge_label_color = edge_label_color
        self.pen_width = pen_width
        self.background_color = background_color
        self.dpi = dpi
        self.rotate = rotate
        self.add_legend = add_legend

    def export(self):

        labels = self.tree.labels_
        self.num_labels = len(labels)

        # esablish the color scheme
        color_brew = _color_brew(self.num_labels, alpha=1.5)
        self.class_colors = {labels[i]: color_brew[i] for i in range(self.num_labels)}

        nodes = get_nodes(
            self.tree.root, 
            get_leaves=True, 
            get_non_leaves=True)

        for i, node in enumerate(nodes):
            node.unique_id = i

        with open(self.outfile, 'w+') as f:
            self._write_tree_attributes(f)
            self._write_node_attributes(f)
            self._write_edge_attributes(f)

            leaf_node_ids = []
            final_string = "{rank = same;" + ';'.join(leaf_node_ids) + "}" 
            
            if self.add_legend:
                final_string += "{rank = same; LEGEND;1;}"

            final_string += '}'

            f.write(final_string)

        for i, node in enumerate(nodes):
            delattr(node, 'unique_id')

    def _write_tree_attributes(self, f):
        f.write('graph Tree {\n')
        f.write('node [shape=box')

        style = ['filled', 'rounded']
        style = ', '.join(style)

        f.write(''', style="{}",color="{}",penwidth="{}",fontcolor="{}", \
                fontname=helvetica] ;\n'''.format(style,
                                                  self.text_color,
                                                  self.pen_width,
                                                  self.text_color))

        f.write('''graph [ranksep="0 equally", splines=straight, \
                bgcolor={}, dpi={}] ;\n'''.format(self.background_color, self.dpi))

        f.write('edge [fontname=helvetica, color={}] ;\n'.format(
            self.edge_color))

        if self.rotate:
            f.write('rankdir=LR ;\n')

        if self.add_legend:
            self._write_legend(f)


    def _write_legend(self, f):
        label = ''

        if self.response_name is not None:
            label += "Response: {}\n".format(self.response_name)

        label += "Classes: {}\n".format(' | '.join(self.tree.labels_))

        f.write('LEGEND [label="{}",shape=note,align=left,\
                style=filled,fillcolor="slategray",\
                fontcolor="white"];\n'.format(label))

    def _find_node_color(self, node):
        label_freq = node.label_frequency
        total = float(sum(label_freq.values()))
        coloring = np.zeros(3)
        for val, freq in label_freq.items():
            coloring += freq * self.class_colors[val]

        coloring = (coloring / total).astype(int)

        return _rgb_to_hex(coloring)

    def _write_node_attribute(self, f, node):

        node_id = node.unique_id

        node_labels = ''

        if (node.left is None) and (node.right is None):
            max_index = np.argmax(node.value)
            max_val = round(node.value[max_index], 3)
            occurence = sum(node.label_frequency.values()) / self._total_samples
            occurence = round(occurence, 3)
            prediction = self.tree.labels_[max_index]
            
            node_labels += '{}\nProb: {}\nFrac: {}'.format(
                prediction, 
                max_val,
                occurence)
        else:
            if self.feature_names is not None:
                node_labels += self.feature_names[node.col]
            pval = scientific_notation(node.col_pval)
            # node_labels += " pval: {}".format(pval)

        node_color = self._find_node_color(node)
        f.write('{} [label="{}"'.format(node_id, node_labels))
        f.write(', fillcolor="{}"] ;\n'.format(
            node_color))

    def _write_node_attributes(self, f):
        nodes = get_nodes(
            self.tree.root, 
            get_leaves=True, 
            get_non_leaves=True)

        for node in nodes:
            self._write_node_attribute(f, node)

    def _write_edge_labels(self, f, parent_node, node, node_type):

        string_format = '{} -- {} [label="{}",penwidth={}] ;\n'

        parent_id = parent_node.unique_id

        if node_type == 'left':
            threshold = parent_node.lthreshold
        elif node_type == 'right':
            threshold = parent_node.rthreshold
        else:
            raise ValueError

        child_id = node.unique_id
        edge_label = ' ' + '\\n '.join(threshold)
        
        f.write(string_format.format(parent_id,
                                     child_id,
                                     edge_label,
                                     self.pen_width))

    def _write_edge_attribute(self, f, node):
        left_node = node.left
        right_node = node.right

        if left_node is not None:
            self._write_edge_labels(f, node, left_node, 'left')
        if right_node is not None:
            self._write_edge_labels(f, node, right_node, 'right')

    def _write_edge_attributes(self, f):
        nodes = get_nodes(
            self.tree.root, 
            get_leaves=True, 
            get_non_leaves=True)

        for node in nodes:
            self._write_edge_attribute(f, node)



