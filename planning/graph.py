#%%
import networkx as nx
import matplotlib.pyplot as plt
import dill
import os
from utils import load_config
from tarski.search import GroundForwardSearchModel
from tarski.model import Model
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators

def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center

def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    attribute : str, default None
        If non-None, the position of each node will be stored on the graph as
        an attribute named `attribute` which can be accessed with
        `G.nodes[...][attribute]`. The function still returns the dictionary.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    See Also
    --------
    rescale_layout_dict
    """
    import numpy as np

    # Find max length over all dimensions
    pos -= pos.mean(axis=0)
    lim = np.abs(pos).max()  # max coordinate for all axes
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        pos *= scale / lim
    return pos

def multipartite_layout(
    G, subset_key="subset", align="vertical", scale=1, center=None, store_pos_as=None
):
    """Position nodes in layers of straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    subset_key : string or dict (default='subset')
        If a string, the key of node data in G that holds the node subset.
        If a dict, keyed by layer number to the nodes in that layer/subset.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    store_pos_as : str, default None
        If non-None, the position of each node will be stored on the graph as
        an attribute with this string as its name, which can be accessed with
        ``G.nodes[...][store_pos_as]``. The function still returns the dictionary.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> pos = nx.multipartite_layout(G)
    >>> # suppress the returned dict and store on the graph directly
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> _ = nx.multipartite_layout(G, store_pos_as="pos")

    or use a dict to provide the layers of the layout

    >>> G = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
    >>> layers = {"a": [0], "b": [1], "c": [2, 3], "d": [4]}
    >>> pos = nx.multipartite_layout(G, subset_key=layers)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    Network does not need to be a complete multipartite graph. As long as nodes
    have subset_key data, they will be placed in the corresponding layers.

    """
    import numpy as np

    if align not in ("vertical", "horizontal"):
        msg = "align must be either vertical or horizontal."
        raise ValueError(msg)

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    try:
        # check if subset_key is dict-like
        if len(G) != sum(len(nodes) for nodes in subset_key.values()):
            raise nx.NetworkXError(
                "all nodes must be in one subset of `subset_key` dict"
            )
    except AttributeError:
        # subset_key is not a dict, hence a string
        node_to_subset = nx.get_node_attributes(G, subset_key)
        if len(node_to_subset) != len(G):
            raise nx.NetworkXError(
                f"all nodes need a subset_key attribute: {subset_key}"
            )
        subset_key = nx.utils.groups(node_to_subset)

    # Sort by layer, if possible
    try:
        layers = dict(sorted(subset_key.items()))
    except TypeError:
        layers = subset_key

    pos = None
    nodes = []
    width = len(layers)
    for i, layer in enumerate(layers.values()):
        height = len(layer)
        xs = np.repeat(i, height)
        ys = np.arange(0, height, dtype=float)
        offset = ((width - 1) / 2, (height - 1) / 2)
        layer_pos = np.column_stack([xs, ys]) - offset
        if pos is None:
            pos = layer_pos
        else:
            pos = np.concatenate([pos, layer_pos])
        nodes.extend(layer)
    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))

    if store_pos_as is not None:
        nx.set_node_attributes(G, pos, store_pos_as)

    return pos

def bfs_layers(G, sources):
    """Returns an iterator of all the layers in breadth-first search traversal.

    Parameters
    ----------
    G : NetworkX graph
        A graph over which to find the layers using breadth-first search.

    sources : node in `G` or list of nodes in `G`
        Specify starting nodes for single source or multiple sources breadth-first search

    Yields
    ------
    layer: list of nodes
        Yields list of nodes at the same distance from sources

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> dict(enumerate(nx.bfs_layers(G, [0, 4])))
    {0: [0, 4], 1: [1, 3], 2: [2]}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> dict(enumerate(nx.bfs_layers(H, [1])))
    {0: [1], 1: [0, 3, 4], 2: [2], 3: [5, 6]}
    >>> dict(enumerate(nx.bfs_layers(H, [1, 6])))
    {0: [1, 6], 1: [0, 3, 4, 2], 2: [5]}
    """

    for node in G:
        if node.action is None:
            sources = node
            break

    sources = [sources]

    current_layer = list(sources)
    visited = set(sources)

    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    while current_layer:
        yield current_layer
        next_layer = []
        for node in current_layer:
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    next_layer.append(child)
        current_layer = next_layer

def bfs_layout(G, start, *, align="vertical", scale=1, center=None, store_pos_as=None):
    """Position nodes according to breadth-first search algorithm.

    Parameters
    ----------
    G : NetworkX graph
        A position will be assigned to every node in G.

    start : node in `G`
        Starting node for bfs

    center : array-like or None
        Coordinate pair around which to center the layout.

    store_pos_as : str, default None
        If non-None, the position of each node will be stored on the graph as
        an attribute with this string as its name, which can be accessed with
        ``G.nodes[...][store_pos_as]``. The function still returns the dictionary.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> from pprint import pprint
    >>> G = nx.path_graph(4)
    >>> pos = nx.bfs_layout(G, 0)
    >>> # suppress the returned dict and store on the graph directly
    >>> _ = nx.bfs_layout(G, 0, store_pos_as="pos")
    >>> pprint(nx.get_node_attributes(G, "pos"))
    {0: array([-1.,  0.]),
     1: array([-0.33333333,  0.        ]),
     2: array([0.33333333, 0.        ]),
     3: array([1., 0.])}



    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    G, center = _process_params(G, center, 2)

    # Compute layers with BFS
    layers = dict(enumerate(bfs_layers(G, start)))

    if len(G) != sum(len(nodes) for nodes in layers.values()):
        raise nx.NetworkXError(
            "bfs_layout didn't include all nodes. Perhaps use input graph:\n"
            "        G.subgraph(nx.node_connected_component(G, start))"
        )

    # Compute node positions with multipartite_layout
    pos = multipartite_layout(
        G, subset_key=layers, align=align, scale=scale, center=center
    )

    if store_pos_as is not None:
        nx.set_node_attributes(G, pos, store_pos_as)

    return pos
#%% import the .pkl file
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# domain = 'coffee'
# iter = 0
# graph_file = os.path.join(project_root, 'planning', 'PDDL', domain, 'graph_1.pkl')
# with open(graph_file, 'rb') as f:
#     G = dill.load(f)



# #%% visualize the graph
# col = [G.nodes[n]['color'] if 'color' in G.nodes[n] else 'lightblue' for n in G.nodes()]
# pos = bfs_layout(G, start="None")
# plt.clf()  # clear the current figure
# # make the nodes
# sizes = [G.nodes[n]['size'] + 700 if 'size' in G.nodes[n] else 1000 for n in G.nodes()]
# plt.figure(figsize=(10, 10))
# nx.draw(G, pos, with_labels=True, node_color=col, font_size=5, font_color='black', edge_color=[d['color'] for u, v, d in G.edges(data=True)], width=[d['weight'] for u, v, d in G.edges(data=True)], labels={n: n.action.name[:8] if n.action else "None" for n in G.nodes()})
# graph_image_file_name = project_root + os.sep + 'planning' + os.sep + f"graph_0.png"
# plt.show()
# plt.savefig(graph_image_file_name, format='png', bbox_inches='tight')
                
# %%
