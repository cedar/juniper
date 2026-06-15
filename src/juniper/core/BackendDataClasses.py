from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from typing import Callable

from .Element import Element
from .Circuit import Circuit

import numpy as np


"""
Central place to define dataclasses and types that are used by the backend Engine, Compiler and RuntimState. 
Here we have a central place where we can look up types and document them properly.
"""



"""Types"""
StateTree = dict[str, Any]
KernelMap = dict[str, Callable]
Path = tuple[str, ...]


TimingInfo = dict[str, Any]
Recording = list[list[np.ndarray]]




"""DataClasses"""
@dataclass(frozen=True)
class ElementRef:
    """Reference to an element inside the runtime state. Include nice to have path representations."""

    element: Element

    @property
    def name(self) -> str:
        """Return the connectable name."""
        return self.element.get_local_circuit_id()
    
    @property
    def path_str(self) -> str:
        """Return the path of the element in string format. (i.e. 'circ0.field1')"""
        path_str = ""
        for sub_str in self.path:
            path_str += sub_str + "."
        return path_str[:-1]

    @property     
    def path_objs(self) -> tuple[Element, ...]:
        """Return the elemnt objects in the elements path. Including the Ref element: (circ0, sub_circ, field0)"""
        path_objs = (self.element,)
        for sub_str in reversed(self.path[:-1]):
            path_objs = (self.element.parent, ) + path_objs
        return path_objs 
    
    @property
    def path(self) -> Path:
        """Return the path to the element."""
        return self.element.get_path()



@dataclass
class CompileInfo:
    """Runtime result of compilation.

    This object holds the compiled information of a circuit and is used by Engine. For graph traversal,
    state location, kernels, IO endpoints, and process ownership are gathered
    here so Engine does not need to inspect the user-defined circuit object directly.

    shape:
        compiled_elements[("sub", "field")] -> ElementRef(path, field)
        sources/sinks/dynamic/static -> lists of ElementRef
        kernel_map -> nested tree that mirrors Circuit.compute_kernel
    """

    circuit: Circuit
    compiled_elements: dict[Path, ElementRef]
    dynamic: list[ElementRef]
    static: list[ElementRef]
    sources: list[ElementRef]
    sinks: list[ElementRef]
    kernel_map: KernelMap

    def ref_at(self, path: Path) -> ElementRef:
        """Look up the compiled element ref for a path."""
        return self.compiled_elements[path]

    def dynamic_step_paths(self) -> set[Path]:
        """Return paths to dynamic steps that need fresh PRNG keys."""
        return {
            ref.path
            for ref in self.dynamic
            if not isinstance(ref.element, Circuit)
        }

    def gather_connections(self) -> list[ElementRef]:
        """Return refs for source and sink. used for open/close connections."""
        refs = []
        seen = set()
        for group in (self.sources, self.sinks):
            for ref in group:
                element_id = id(ref.element)
                if element_id in seen:
                    continue
                seen.add(element_id)
                refs.append(ref)
        return refs
