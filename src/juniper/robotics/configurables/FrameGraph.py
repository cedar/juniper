from ...configurables import Configurable
from .Transform import Transform
from collections import deque

import jax.numpy as jnp
from collections import defaultdict, deque

def find_path(edges, start, goal): 
    """
    edges: {(u, v): T} meaning T maps u -> v
    start, goal: nodes
    invert: function returning the inverse transform for a given T

    Returns (nodes_path, transforms_path) where:
      - nodes_path is [start, ..., goal]
      - transforms_path is [T0, T1, ...] applied in order along the path
    If no path exists, returns (None, None).
    """

    # Build adjacency with implicit reverse edges via invert(T)
    adj = defaultdict(list)
    for (u, v), T in edges.items():
        adj[u].append((v, T))            # forward u->v uses T

    # BFS
    q = deque([start])
    parent = {start: None}
    edge_transform_to = {}

    while q:
        node = q.popleft()
        if node == goal:
            break
        for nbr, Tstep in adj.get(node, []):
            if nbr not in parent:
                parent[nbr] = node
                edge_transform_to[nbr] = Tstep
                q.append(nbr)

    if goal not in parent:
        return None, None

    # Reconstruct nodes [start..goal]
    nodes_path = []
    cur = goal
    while cur is not None:
        nodes_path.append(cur)
        cur = parent[cur]
    nodes_path.reverse()

    # Reconstruct transforms in forward order
    transforms_rev = []
    cur = goal
    while parent[cur] is not None:
        transforms_rev.append(edge_transform_to[cur])
        cur = parent[cur]
    transforms_path = list(reversed(transforms_rev))

    return nodes_path, transforms_path

class FrameGraph(Configurable):
    """
    Description
    ---------
    A wrapper object to construct a frame graph. The frame graph is a directed graph with coordinate frames as vertices and transformations between frames as edges.
    New frames and transformations can be added by calling the add_edge() function. Once all frames are specified the frame graph can be used to find a 'path' from 
    a source frame to a target frame. Individual transformations will automatically be chained (and inverted if necessary) to produce the transformation from any 
    source to any reachable target frame.

    """
    def __init__(self, params):
        mandatory_params = []
        super().__init__(params, mandatory_params)

        if "edges" not in params:
            self.edges = {}
        else:
            self.edges = params["edges"]

    def add_edge(self, source : str, target : str, transform : Transform):
        """
        Description
        ---------
        Adds an edge to the frame graph. Source and target frame do not already need to be known. The edge is given in the form of a Transform objbect.

        """
        self.edges[(source, target)] = transform
        if (source, target) not in self.edges.keys():
            self.edges[(source, target)] = transform
        elif (target, source) not in self.edges.keys():
            self.edges[(target, source)] = Transform(params={"M_func": lambda joint_state: jnp.linalg.inv(transform.M_func(joint_state))})

    def lookup(self, source, target):
        if target == source:
            return Transform(params={"M_func": lambda joint_state: jnp.eye(4)})
        elif (source, target) in self.edges.keys():
            return self.edges[(source,target)]
        elif (target, source) in self.edges.keys():
            inv_trans = self.edges[(target, source)]
            trans = Transform(params={"M_func": lambda joint_state: jnp.linalg.inv(inv_trans.M_func(joint_state))})
            self.add_edge(source, target, trans)
            return self.edges[(source, target)]
        else:
            # find a path
            _, transform_path = find_path(self.edges, source, target)

            def composed_transform(joint_angles, transform_path):
                comb_trans = jnp.eye(4)
                for trans in reversed(transform_path):
                    comb_trans = comb_trans @ trans.M_func(joint_angles)
                return comb_trans
            
            combined_transform = Transform(params={"M_func": lambda joint_angles: composed_transform(joint_angles, transform_path)})

            self.add_edge(source, target, combined_transform)

            return self.edges[(source, target)]


