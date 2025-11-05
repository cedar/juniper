from .Configurable import Configurable
from .Transform import Transform
#from .Transform import ComposedTransform
from collections import deque

import jax.numpy as jnp
import jax.debug as jdbg

class FrameGraph(Configurable):
    def __init__(self, params):
        mandatory_params = []
        super().__init__(params, mandatory_params)

        if "edges" not in params:
            self.edges = {}
        else:
            self.edges = params["edges"]

        self.composed_edges = {}

    def add_edge(self, source, target, Transform):
        self.edges[(source, target)] = Transform

    def lookup(self, source, target):
        if target == source:
            return Transform(params={"M_func": lambda joint_state: jnp.eye(4)})
        
        visited_sources = []
        visited_targets = []
        for edge, transform in self.edges.items():
            #jdbg.print("sheesh")
            #jdbg.print("{}", self.edges)
            #jdbg.print("{}", edge)
            #jdbg.print("{}", (source, target)==edge)
            visited_sources += [edge[0]]
            visited_targets += [edge[1]]
            
            if (source, target) == edge:
                return transform
            elif (target, source) in self.edges:
                self.edges[(source, target)] = Transform(params={"M_func": lambda joint_state: jnp.linalg.inv(transform.M_func(joint_state))})
                return self.edges[(source, target)]
            
        """# find path
        if (source, target) in self.composed_edges:
            return self.composed_edges[(source, target)]
        else:
            edges = self.edges
            neighbors = {} 
            for (p, c), T in edges.items():
                neighbors.setdefault(p, []).append((c, True, T))
                neighbors.setdefault(c, []).append((p, False, T))

            prev = {}
            q = deque([source])
            seen = {source}
            while q:
                f = q.popleft()
                for nb, fwd, Tedge in neighbors.get(f, []):
                    if nb in seen: continue
                    prev[nb] = (f, fwd, Tedge)
                    if nb == target:
                        q.clear(); break
                    seen.add(nb); q.append(nb)
            if target not in prev:
                raise RuntimeError(f"Kein Pfad von '{source}' nach '{target}'.")
            
            # Pfad als Liste von Einzelschritten (source → target)
            func_list = []                      # Liste der Teil-Transforms
            cur = target
            while cur != source:
                p, fwd, Tedge = prev[cur]
                M_step = Tedge.M_func if fwd else lambda joint_state: jnp.linalg.inv(Tedge.M_func(joint_state))
                func_list.insert(0, M_step)    
                cur = p
            
            self.composed_edges[(source, target)] = ComposedTransform(params={"M_func": func_list})
            return self.composed_edges[(source, target)]"""
