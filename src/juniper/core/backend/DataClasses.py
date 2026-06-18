from __future__ import annotations
from dataclasses import dataclass

from typing import Any
from typing import Callable
from typing import Union
from typing import TypeAlias

from ..frontend.Element import Element
from ..frontend.Circuit import Circuit
from ..frontend.Slot import Slot
from ..frontend.Buffer import Buffer
from .Exceptions import RecordingError

import numpy as np
import time
import os
import json
import pickle

"""
Central place to define dataclasses and types that are used by the backend Engine, Compiler and RuntimState. 
Here we have a central place where we can look up types and document them properly.
"""



"""Types"""
KernelMap = dict[str, Callable]
ElementPath = tuple[str, ...]
StateTree = dict[ElementPath, dict[str, Any]]


TimingInfo = dict[str, Any]
RecKey : TypeAlias = Union[str, Element, Buffer, Slot]

"""DataClasses"""
@dataclass(frozen=True)
class ElementRef:
    """Reference to an element inside the runtime state. Include nice-to-have path representations."""

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
    def path(self) -> ElementPath:
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
    compiled_elements: dict[ElementPath, ElementRef]
    dynamic: list[ElementRef]
    static: list[ElementRef]
    sources: list[ElementRef]
    sinks: list[ElementRef]
    kernel_map: KernelMap

    def ref_at(self, path: ElementPath) -> ElementRef:
        """Look up the compiled element ref for a path."""
        return self.compiled_elements[path]

    def dynamic_step_paths(self) -> set[ElementPath]:
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


@dataclass(frozen=True)
class Recording:
    """
    Recorded data of a simulation run. This object holds the raw lists of recorded arrays
    and provides utility functions for easy access, plotting and storage.
    """

    recording : list[list[np.ndarray]]
    keys : list[RecKey]

    @property
    def key_strings(self) -> list[str]:
        """Return the recording target keys as strings."""
        key_strings = []
        for key in self.keys:
            if isinstance(key, str):
                key_strings.append(key)
            else:
                key_strings.append(key.get_path_str())
        return key_strings
    
    def get_key_idx(self, key : RecKey) -> int:
        """Get the idx position of a key."""
        if isinstance(key, str):
            key_str = key
        else:
            key_str = key.get_path_str()
        return self.key_strings.index(key_str)
            
    def get_at_element(self, key : RecKey) -> Recording:
        """Get the recording of a specific element."""
        key_idx = self.get_key_idx(key)
        return Recording([[step_recording[key_idx]] for step_recording in self.recording], [key])
    
    def get_at_elements(self, keys : list[RecKey]) -> Recording:
        """Get the recordings of a list of elements."""
        key_indices = [self.get_key_idx(key) for key in keys]
        return Recording(
            [[step_recording[key_idx] for key_idx in key_indices] for step_recording in self.recording],
            keys,
        )
    
    def get_at_step(self, step_idx : int) -> Recording:
        """Get the recording of all elements at a sepcific time point."""
        return Recording([self.recording[step_idx]], self.keys)

    def get_in_step_interval(self, idx_interval : tuple[int]) -> Recording:
        """Get the recording of all elements inside a specific recording interval."""
        return Recording(self.recording[idx_interval[0]:idx_interval[1]], self.keys)

    def slice(self, keys : RecKey, idx_interval : tuple[int]) -> Recording:
        """Get the recording of specific elements inside a specific recording interval."""
        element_recording = self.get_at_elements(keys)
        return element_recording.get_in_step_interval(idx_interval)
    
    def append(self, recording : Recording):
        """Appends the data of another recording. The recording has to have identical keys"""
        key_strings = recording.key_strings
        if not self.key_strings == key_strings:
            raise RecordingError(f"Can't append recording when recorded steps do not match. {self.key_strings}")

        self.recording.append(recording.recording)

    @classmethod
    def load_from_file(cls, run_dir: str) -> Recording:
        """Load a recording saved by save_to_file."""
        manifest_path = os.path.join(run_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No recording manifest found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        key_strings = list(manifest.get("names", []))
        num_steps = int(manifest.get("num_steps", 0))
        recording = []

        for step_idx in range(num_steps):
            step_path = os.path.join(run_dir, f"t_{step_idx:06d}.pkl")
            if not os.path.exists(step_path):
                raise FileNotFoundError(f"Missing recording step file {step_path}")

            with open(step_path, "rb") as f:
                step_payload = pickle.load(f)

            step_names = list(step_payload.get("names", []))
            if step_names != key_strings:
                raise ValueError(f"Recording step {step_idx} has different keys than manifest")

            step_data = step_payload.get("data", {})
            recording.append([np.asarray(step_data[key]) for key in key_strings])

        return cls(recording=recording, keys=key_strings)

    def save_to_file(self, path: str, run_dir: str = None) -> str:
        """
        Batch writer: 
        - store each timestep of the recording as its own file inside a per-run folder.
        - if previous data exists in folder, the new batch is appended.

        
        Folder layout:
        <path>/run_<timestamp>_<ms>/
            manifest.json
            t_000000.pkl
            t_000001.pkl
            ...

        Returns:
        run_dir: The resolved run folder path to reuse on subsequent writes.
        """
        key_strings = self.key_strings

        first_row = _time_step_to_row(self.recording[0])
        N = len(first_row)
        T = len(self.recording)

        if len(key_strings) != N:
            raise ValueError(f"len(keys)={len(key_strings)} but each recording has N={N} arrays")

        run_dir = _make_run_recording_dir(path, run_dir)

        manifest_path = os.path.join(run_dir, "manifest.json")
        manifest = _make_recording_manifest(manifest_path, key_strings)

        start_t = int(manifest.get("num_steps", 0))

        for t in range(T):
            row = _time_step_to_row(self.recording[t])
            if len(row) != N:
                raise ValueError(f"Inconsistent N at batch step {t}: expected {N}, got {len(row)}")

            step_idx = start_t + t
            step_payload = {
                "t": step_idx,
                "recorded_at_unix": time.time(),
                "names": list(key_strings),
                "data": {key_strings[i]: np.asarray(row[i]) for i in range(N)},
            }
            step_path = os.path.join(run_dir, f"t_{step_idx:06d}.pkl")
            tmp_step_path = step_path + ".tmp"
            with open(tmp_step_path, "wb") as f:
                pickle.dump(step_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_step_path, step_path)

        manifest["num_steps"] = start_t + T
        manifest["last_write_unix"] = time.time()
        tmp_manifest = manifest_path + ".tmp"

        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_manifest, manifest_path)

        return run_dir

    def plot(
        self,
        keys: list[RecKey] = None,
        idx_interval: tuple[int, int] = None,
        time_axis=None,
        snapshot_indices=None,
        group_keys: list[list[RecKey]] = None,
        figsize=(10, 4),
    ):
        """Plot this recording using juniper.util.plotting.plot_steps."""
        from ...util.plotting import plot_steps

        recording = self
        if keys is not None:
            recording = recording.get_at_elements(keys)
        if idx_interval is not None:
            recording = recording.get_in_step_interval(idx_interval)

        return plot_steps(
            recording,
            time_axis=time_axis,
            snapshot_indices=snapshot_indices,
            scalar_group_keys=group_keys,
            figsize=figsize,
        )


def _make_run_recording_dir(base_path : str, run_dir : str) -> str:
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    os.makedirs(base_path, exist_ok=True)

    # Create a unique folder for this simulation run.
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    ms = int((time.time() % 1) * 1000)
    run_dir = os.path.join(base_path, f"run_{ts}_{ms:03d}")
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(base_path, f"run_{ts}_{ms:03d}_{suffix:02d}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def _time_step_to_row(step):
    if isinstance(step, (list, tuple)) and len(step) == 1 and isinstance(step[0], (list, tuple)):
        return step[0]
    return step

def _make_recording_manifest(manifest_path : str, key_strings : list[str]) -> dict[Any]:
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        existing_names = manifest.get("names", [])
        if list(existing_names) != list(key_strings):
            raise ValueError("Existing pickle recording folder has different names; refuse to append.")
    else:
        manifest = {
            "format": "simulation_handler_pickle_per_step_v1",
            "names": list(key_strings),
            "num_steps": 0,
            "created_at_unix": time.time(),
        }
    return manifest
