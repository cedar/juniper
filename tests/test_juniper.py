import functools
import json
import os
import shutil
import tempfile
import juniper as jp
import numpy as np
import pytest
import time
from contextlib import contextmanager
import io
import sys

from juniper.core.backend.Exceptions import JuniperError
from juniper.core.backend.DataClasses import Recording
from juniper.core.frontend import CircuitContext
from juniper.util import util_jax


def clean_arch(arch):
    """Clean architecture test state, including dynamic element attributes."""
    arch.clean()
    arch.engine.clean()
    CircuitContext.set_current(arch)

def function_test(func):
    """Decorator to clean arch after every test."""
    functools.wraps(func)
    def wrapper(*args, **kwargs):
        arch = jp.get_arch()
        clean_arch(arch)
        try:
            func(*args, **kwargs)
        except Exception as e:
            #traceback.print_exc()
            raise JuniperError(f"test_function {func.__name__} failed with the above stderr call") from e
        finally:
            arch = jp.get_arch()
            clean_arch(arch)
    return wrapper

@contextmanager
def simulate_user_input(input : str):
    """Overwrites stdin with synthetic user input. This is only used for pytest for automatic testing."""
    orig = sys.stdin
    sys.stdin = io.StringIO(input + "\n")
    try:
        yield
    finally:
        sys.stdin = orig

def recorded_array(recording, key, step_idx=-1):
    """Return one recorded array via the Recording accessors."""
    return recording.get_at_element(key).get_at_step(step_idx).recording[0][0]

def recorded_step(recording, step_idx=-1):
    """Return all recorded arrays for one simulation step via the Recording accessor."""
    return recording.get_at_step(step_idx).recording[0]

class TestJuniper:
    """
    A simple test class for pytesting. The class owns the architecture singleton, 
    while each test function defines a small test circuit tailord to a specific test case.
    """
    arch = jp.get_arch(name="test_arch")


    @function_test
    def test_recording(self):
        in1 = jp.CustomInput("in1", (1,))
        in1.set_data(np.ones((1,)))

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        recording, timing = self.arch.run_simulation(num_steps=42, steps_to_record=["in1", in1, in1.out0], print_timing=False, save_buffer=False)
        full_recording = recording.slice(["in1", in1, in1.out0], (0, 42))
        assert len(full_recording.recording) == 42
        assert np.isclose(recorded_array(recording, "in1", 0), 1)
        assert np.isclose(recorded_array(recording, in1, 0), 1)
        assert np.isclose(recorded_array(recording, in1.out0, 0), 1)
        assert np.array_equal(np.shape(full_recording.recording), (42,3,1))
        assert np.isclose(np.sum(full_recording.recording), 126)

    @function_test
    def test_recording_save_and_load(self):
        """Test saving and loading of recording."""
        recording = Recording(
            recording=[
                [np.array([1.0], dtype=np.float32), np.array([[1, 2], [3, 4]], dtype=np.float32)],
                [np.array([2.0], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32)],
            ],
            keys=["scalar", "matrix"],
        )
        temp_dir = tempfile.mkdtemp(prefix="juniper_recording_test_")

        try:
            run_dir = recording.save_to_file(temp_dir)
            loaded = Recording.load_from_file(run_dir)

            assert loaded.key_strings == recording.key_strings
            assert len(loaded.recording) == len(recording.recording)
            for step_idx in range(len(recording.recording)):
                for key in recording.key_strings:
                    assert np.array_equal(
                        recorded_array(loaded, key, step_idx),
                        recorded_array(recording, key, step_idx),
                    )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @function_test
    def test_recording_plot(self):
        """Test plotting of a recording."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        recording = Recording(
            recording=[
                [np.array([0.0], dtype=np.float32), np.arange(4, dtype=np.float32).reshape(2, 2)],
                [np.array([1.0], dtype=np.float32), np.ones((2, 2), dtype=np.float32)],
            ],
            keys=["scalar", "matrix"],
        )

        fig = recording.plot(
            keys=["scalar", "matrix"],
            idx_interval=(0, 2),
            snapshot_indices=[1],
            figsize=(3, 2),
        )
        try:
            assert fig is not None
            assert len(fig.axes) >= 2
        finally:
            plt.close(fig)

    @function_test
    def test_custom_input_updates_after_compile(self):
        """Sources should push new CPU-side data into runtime state each tick."""
        in1 = jp.CustomInput("in1", (2,))
        in1.set_data(np.array([1, 2], dtype=np.float32))

        self.arch.compile(warmup=1, print_compile_info=False, load_buffer=False)
        recording1, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["in1"], print_timing=False)

        in1.set_data(np.array([3, 4], dtype=np.float32))
        recording2, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["in1"], print_timing=False)

        assert np.allclose(recorded_array(recording1, "in1"), np.array([1, 2], dtype=np.float32))
        assert np.allclose(recorded_array(recording2, "in1"), np.array([3, 4], dtype=np.float32))


    @function_test
    def test_static_arch(self):
        """test: 1*3+1=4"""
        const = jp.AddConstant("constant", 2)
        in1 = jp.GaussInput("input", (1,), (0.01,), 1, center=(0.,))
        in2 = jp.GaussInput("input2", (1,), (0.01,), 1, center=(0.,))
        mult = jp.ComponentMultiply("mult")
        out = jp.Sum("out")
        in1 >> const >> mult >> out
        in2 >> out

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        recording, timing = self.arch.run_simulation(num_steps=3, steps_to_record=["out"], print_timing=False, save_buffer=False)

        last_out = recorded_array(recording, "out")
        res = np.sum(last_out)

        assert np.isclose(res, 4)

    @function_test
    def test_juniper_syntax(self):
        """test all syntactic categories implemented in the juniper dsl"""
        const = jp.AddConstant("constant", 2)
        in1 = jp.GaussInput("input", (1,), (0.01,), 1, center=(0.,))
        in2 = jp.GaussInput("input2", (1,), (0.01,), 1, center=(0.,))
        mult = jp.ComponentMultiply("mult")
        out = jp.Sum("out")
        out2 = jp.Sum("out2")
        # (1+2)*1+1=4
        # 4+3=7
        in1 >> const
        in2.out0 >> out >> out2
        in2 >> "mult"
        const.out0 >> "mult.in0"
        out.in0 << "mult"
        out2 << mult.out0

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        recording, timing = self.arch.run_simulation(num_steps=5, steps_to_record=["out", "out2"], print_timing=False, save_buffer=False)

        last_out = recorded_array(recording, "out")
        last_out2 = recorded_array(recording, "out2")

        res = np.sum(last_out)
        res2 = np.sum(last_out2)

        assert np.isclose(res, 4)
        assert np.isclose(res2, 7)

    @function_test
    def test_nested_circuit(self):
        """Tests if nested circuits are handled correctly."""
        circ = jp.Circuit("circ")
        with circ as c:
            summed_inputs = jp.Sum("summed_inputs")
            c.register_input_slot("in0")
            c.register_input_slot("in1")
            c.register_input_slot("in2")
            c.register_input_slot("in3")
            c.register_output_slot("out0")
            
            c.in0 >> summed_inputs
            c.in1 >> summed_inputs
            c.in2 >> summed_inputs
            c.in3 >> summed_inputs >> c.out0
            

        circ2 = jp.Circuit("circ2")
        with circ2 as c:
            summed_inputs = jp.Sum("summed_inputs")
            c.register_input_slot("in0")
            c.register_output_slot("out0")
            c.in0 >> summed_inputs >> c.out0

        external_input = jp.CustomInput("external_input", (1,))
        external_input.set_data(np.ones((1,)))

        external_input >> circ.in0
        external_input >> circ.in1
        external_input >> circ.in2
        external_input >> circ.in3

        circ >> circ2

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        self.arch.reset_state()
        recording, timing = self.arch.run_simulation(num_steps=1, steps_to_record=[circ, "circ2"], print_timing=False, save_buffer=False)

        last_out = recorded_array(recording, circ)
        last_out2 = recorded_array(recording, "circ2")

        res = np.sum(last_out)
        res2 = np.sum(last_out2)

        assert np.isclose(res, 4)
        assert np.isclose(res2, 4)

    @function_test
    def test_compile_info_flattened_nested_circuit(self):
        """Top-level CompileInfo should expose nested elements with extended paths."""
        circ = jp.Circuit("circ")
        with circ as c:
            summed_inputs = jp.Sum("summed_inputs")
            c.register_input_slot("in0")
            c.register_output_slot("out0")
            c.in0 >> summed_inputs >> c.out0

        external_input = jp.CustomInput("external_input", (1,))
        external_input.set_data(np.ones((1,)))
        external_input >> circ

        self.arch.compile(warmup=0, print_compile_info=False, load_buffer=False)

        compile_info = self.arch.engine.compile_info
        assert ("circ",) in compile_info.compiled_elements
        assert ("circ", "summed_inputs") in compile_info.compiled_elements
        assert compile_info.compiled_elements[("circ", "summed_inputs")].element.get_local_circuit_id() == "summed_inputs"
        assert ("circ",) in compile_info.kernel_map
        assert ("circ", "summed_inputs") in compile_info.kernel_map

    @function_test
    def test_component_multiply_product_aggregation(self):
        """ComponentMultiply should multiply multiple incoming values instead of summing them."""
        in1 = jp.CustomInput("in1", (1,))
        in2 = jp.CustomInput("in2", (1,))
        in3 = jp.CustomInput("in3", (1,))
        in1.set_data(np.array([2], dtype=np.float32))
        in2.set_data(np.array([3], dtype=np.float32))
        in3.set_data(np.array([4], dtype=np.float32))
        mult = jp.ComponentMultiply("mult")

        in1 >> mult
        in2 >> mult
        in3 >> mult

        self.arch.compile(warmup=1, print_compile_info=False, load_buffer=False)
        recording, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["mult"], print_timing=False)

        assert np.isclose(np.sum(recorded_array(recording, "mult")), 24)

    @function_test
    def test_duplicate_element_name_fails(self):
        """Circuit construction should reject duplicate element names."""
        jp.CustomInput("in1", (1,))
        with pytest.raises(Exception):
            jp.CustomInput("in1", (1,))

    @function_test
    def test_max_incoming_connection_fails(self):
        """Slots with one allowed incoming connection should reject a second source."""
        in1 = jp.CustomInput("in1", (1,))
        in2 = jp.CustomInput("in2", (1,))
        out = jp.AddConstant("out", 1)
        in1 >> out
        with pytest.raises(Exception):
            in2 >> out

    @function_test
    def test_reset(self):
        """tests if reset correctly restores the initial state and doesn't cause retracing from jax."""
        nf1 = jp.NeuralField("nf1", (50,50), sigmoid="ExpSigmoid", beta=100, theta=0, resting_level=-5, global_inhibition=0, input_noise_gain=0, tau=0.02)
        gauss = jp.GaussInput("gauss", (50,50), (3,3), 6, center=(25,25))
        gauss >> nf1

        self.arch.compile(warmup=0, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        t_comp = time.time()
        compile_recording, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["nf1"], print_timing=False, save_buffer=False)
        t_comp = time.time() - t_comp
        compile_recording, _ = self.arch.run_simulation(num_steps=5, steps_to_record=["nf1"], print_timing=False, save_buffer=False)
        cache_size = getattr(self.arch.engine._tick, "_cache_size", None)
        cache_after_compile = cache_size() if cache_size is not None else None

        self.arch.reset_state()
        t1 = time.time()
        recording1, timing1 = self.arch.run_simulation(num_steps=1, steps_to_record=["nf1"], print_timing=False, save_buffer=False)
        t1 = time.time()-t1
        cache_after_first_reset = cache_size() if cache_size is not None else None

        recording2, _ = self.arch.run_simulation(num_steps=100, steps_to_record=["nf1"], print_timing=False, save_buffer=False)

        self.arch.reset_state()

        t2 = time.time()
        recording3, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["nf1"], print_timing=False, save_buffer=False)
        t2 = time.time() - t2
        cache_after_second_reset = cache_size() if cache_size is not None else None

        nf_1 = recorded_array(recording1, "nf1")
        nf_2 = recorded_array(recording2, "nf1")
        nf_3 = recorded_array(recording3, "nf1")
        res_nf_1 = np.sum(nf_1)
        res_nf_2 = np.sum(nf_2)
        res_nf_3 = np.sum(nf_3)

        assert nf_1.shape == (50, 50)
        assert nf_3.shape == (50, 50)
        assert nf_1.dtype == util_jax.cfg["dtype"]
        assert nf_3.dtype == util_jax.cfg["dtype"]
        assert np.isclose(res_nf_1, res_nf_3)
        assert not np.isclose(res_nf_1, res_nf_2)
        if cache_size is not None:
            assert cache_after_first_reset == cache_after_compile
            assert cache_after_second_reset == cache_after_compile
        else:
            assert abs(t2-t1) < abs(t2-t_comp)

    @function_test
    def test_dnn_prompting(self):
        """Tests DNN download prompting."""

        with pytest.raises(Exception):
            with simulate_user_input("n"):
                jp.DNN("dnn2", "4_3")

    @function_test
    def test_buffer_save_and_load(self):
        """Permanent buffers should be saved to disk and loaded into a fresh state."""
        self.arch.set_arch_name("test_buffer_arch")
        data_file = util_jax.cfg["arch_file_path"] + self.arch.get_local_circuit_id() + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)

        try:
            build_bcm_buffer_circuit()
            self.arch.compile(warmup=0, print_compile_info=False, load_buffer=False)
            self.arch.run_simulation(
                num_steps=1,
                steps_to_record=[],
                print_timing=False,
                save_buffer=True,
            )

            assert os.path.exists(data_file)
            with open(data_file, "r") as f:
                saved_tree = json.load(f)
            assert "bcm" in saved_tree
            assert "wheights" in saved_tree["bcm"]["BUFFER"]
            assert "theta" in saved_tree["bcm"]["BUFFER"]

            saved_tree["bcm"]["BUFFER"]["wheights"] = [[[[0.75]]]]
            saved_tree["bcm"]["BUFFER"]["theta"] = [[[0.25]]]
            with open(data_file, "w") as f:
                json.dump(saved_tree, f)

            clean_arch(self.arch)

            build_bcm_buffer_circuit()
            self.arch.compile(warmup=0, print_compile_info=False, load_buffer=True)

            bcm_ref = self.arch.engine.compile_info.ref_at(("bcm",))
            bcm_state = self.arch.engine.state.get(bcm_ref)
            assert np.allclose(np.array(bcm_state["wheights"]), np.array([[[[0.75]]]], dtype=np.float32))
            assert np.allclose(np.array(bcm_state["theta"]), np.array([[[0.25]]], dtype=np.float32))
        finally:
            if os.path.exists(data_file):
                os.remove(data_file)

    @function_test
    def test_tcp_connection(self):
        """Tests if tcp sockets are handles correctly and establish working connections."""
        tcp_params = {
            "ip": "127.0.0.1",
            "port": 50030,
            "shape": (3,),
            "dtype": np.uint8,
            "send_on_change_only": False,
            "time_step": 0.005,
            "connect_retry_delay": 0.01,
        }
        jp.TCPReader("tcp_reader", **tcp_params)
        tcp_writer = jp.TCPWriter("tcp_writer", **tcp_params)

        input = jp.CustomInput("input", (3,))
        in_array = np.asanyarray([1,2,3], dtype=np.uint8)
        input.set_data(in_array)
        input >> tcp_writer

        out_array = np.zeros_like(in_array)
        try:
            self.arch.compile(warmup=0, print_compile_info=False, load_buffer=False)
            time.sleep(0.2)

            for _ in range(40):
                recording, _ = self.arch.run_simulation(
                    num_steps=1,
                    steps_to_record=["tcp_reader"],
                    print_timing=False,
                    save_buffer=False,
                )
                out_array = np.asanyarray(recorded_array(recording, "tcp_reader"))
                if np.array_equal(out_array.astype(np.uint8), in_array):
                    break
                time.sleep(0.05)
        finally:
            self.arch.close_connections()

        assert np.array_equal(out_array.astype(np.uint8), in_array)

    @function_test
    def test_invalid_shape_merge(self):
        """Tests if the comiler catches invalid input shapes."""

        # case 1
        out = jp.Sum("out")
        in1 = jp.CustomInput("in1", (2,))
        in2 = jp.CustomInput("in2", (2,2))
        in1 >> out
        in2 >> out
        with pytest.raises(Exception):
            self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        clean_arch(self.arch)

        out = jp.NeuralField("out", (50,50), sigmoid="AbsSigmoid", beta=100, theta=0, resting_level=-5, global_inhibition=0, input_noise_gain=0, tau=0.02)
        in1 = jp.CustomInput("in1", (50,50))
        in1 >> out
        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        clean_arch(self.arch)

        # case 2
        out = jp.NeuralField("out", (50,50), sigmoid="AbsSigmoid", beta=100, theta=0, resting_level=-5, global_inhibition=0, input_noise_gain=0, tau=0.02)
        in1 = jp.CustomInput("in1", (50,50))
        in2 = jp.CustomInput("in2", (1,))
        proj = jp.CompressAxes("proj", (0,1), "Maximum")
        in1 >> out
        in2 >> out
        out >> proj >> out
        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        clean_arch(self.arch)

        # case 3
        out = jp.NeuralField("out", (50,50), sigmoid="AbsSigmoid", beta=100, theta=0, resting_level=-5, global_inhibition=0, input_noise_gain=0, tau=0.02)
        in1 = jp.CustomInput("in1", (50,50))
        in2 = jp.CustomInput("in2", (1,))
        proj = jp.CompressAxes("proj", (0,), "Maximum")
        in1 >> out
        in2 >> out
        out >> proj >> out
        with pytest.raises(Exception):
            self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        clean_arch(self.arch)

        # case 4
        out = jp.NeuralField("out", (50,50), sigmoid="AbsSigmoid", beta=100, theta=0, resting_level=-5, global_inhibition=0, input_noise_gain=0, tau=0.02)
        in1 = jp.CustomInput("in1", (50,50))
        in2 = jp.CustomInput("in2", (1,))
        proj = jp.CompressAxes("proj", (0,1), "Maximum", compress_all=True)
        in1 >> out
        in2 >> out
        out >> proj >> out
        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        clean_arch(self.arch)

        # case 5
        out = jp.NeuralField("out", (50,50), sigmoid="AbsSigmoid", beta=100, theta=0, resting_level=-5, global_inhibition=0, input_noise_gain=0, tau=0.02)
        in1 = jp.CustomInput("in1", (50,50))
        in2 = jp.CustomInput("in2", ())
        proj = jp.CompressAxes("proj", (0,1), "Maximum", compress_all=True)
        in1 >> out
        in2 >> out
        out >> proj >> out
        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        clean_arch(self.arch)

    @function_test
    def test_nested_state_update(self):
        """Tests if steps in nested circuits update their state correctly and can be recorded."""
        c = jp.Circuit("c")
        with c as c:
            s = jp.Sum("s")
            c.register_input_slot("in0")
            c.in0 >> s
        
        input = jp.CustomInput("input", (3,))
        in_array = np.asanyarray([1,2,3], dtype=np.float32)
        input.set_data(in_array)

        input >> c

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        
        recording, timing = self.arch.run_simulation(num_steps=1, steps_to_record=["c.s", "c.s.out0"], print_timing=False, save_buffer=False)

        out_array = np.asanyarray(recorded_array(recording, "c.s"))
        explicit_out_array = np.asanyarray(recorded_array(recording, "c.s.out0"))

        assert np.array_equal(in_array, out_array)
        assert np.array_equal(in_array, explicit_out_array)

    @function_test
    def test_circuit_output_multiple_sources(self):
        """Circuit output slots should sum all registered internal source slots."""
        c = jp.Circuit("c")
        with c as c:
            sum0 = jp.Sum("sum0")
            sum1 = jp.Sum("sum1")
            c.register_input_slot("in0")
            c.register_input_slot("in1")
            c.register_output_slot("out0")
            c.in0 >> sum0 >> c.out0
            c.in1 >> sum1 >> c.out0

        input0 = jp.CustomInput("input0", (1,))
        input1 = jp.CustomInput("input1", (1,))
        input0.set_data(np.ones((1,), dtype=np.float32))
        input1.set_data(np.ones((1,), dtype=np.float32) * 2)

        input0 >> c.in0
        input1 >> c.in1

        self.arch.compile(warmup=1, print_compile_info=False, load_buffer=False)
        recording, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["c"], print_timing=False, save_buffer=False)

        assert np.isclose(np.sum(recorded_array(recording, "c")), 3)

    @function_test
    def test_deeply_nested_circuits(self):
        """Tests if nesting of multiple circuits works. Checking state update, recording and connections."""
        sc = jp.Circuit("sc")
        with sc as sc:
            ssc = jp.Circuit("ssc")
            with ssc as ssc:
                sssc = jp.Circuit("sssc")
                with sssc as sssc:
                    inner_state = jp.Sum("inner_state")
                    sssc.register_input_slot("in0")
                    sssc.register_output_slot("out0")
                    sssc >> inner_state >> sssc
                ssc.register_input_slot("in0")
                ssc.register_output_slot("out0")
                ssc >> sssc >> ssc.out0
            sc.register_input_slot("in0")
            sc.register_output_slot("out0")
            sc.in0 >> ssc >> sc
        external_input = jp.CustomInput("external_input", (3,))
        external_input >> sc

        in_array = np.array([1,2,3])
        external_input.set_data(in_array)

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        recording, _ = self.arch.run_simulation(num_steps=10, steps_to_record=[sc, ssc, sssc, inner_state], print_timing=False, save_buffer=False)

        last_step = recorded_step(recording)

        for out_state in last_step:
            out_array = np.asanyarray(out_state)
            assert np.array_equal(in_array, out_array)

    @function_test
    def test_constructor_methods(self):
        """Tests if the constructor methods for creating steps using params or individual args works, and defaults work."""
        params = {"shape":(50,50), "sigmoid":"AbsSigmoid", "beta":100, "theta":0, "resting_level":-5, "global_inhibition":0, "input_noise_gain":0, "tau":0.02}
        jp.NeuralField("_1", (50,50), sigmoid="AbsSigmoid", beta=100, theta=0, resting_level=-5, global_inhibition=0, input_noise_gain=0, tau=0.02)
        jp.NeuralField("_2", **params)
        jp.NeuralField.from_params("_3", params)
        jp.NeuralField("_4", shape=(50,50))
        self.arch._1 >> self.arch._2 >> self.arch._3

        self.arch.compile()
        assert self.arch.is_compiled
        assert self.arch._4._params["sigmoid"] == jp.NeuralField._sigmoid
        assert self.arch._4._params["beta"] == jp.NeuralField._beta
        assert self.arch._4._params["theta"] == jp.NeuralField._theta
        assert self.arch._4._params["resting_level"] == jp.NeuralField._resting_level
        assert self.arch._4._params["global_inhibition"] == jp.NeuralField._global_inhibition
        assert self.arch._4._params["input_noise_gain"] == jp.NeuralField._input_noise_gain
        assert self.arch._4._params["tau"] == jp.NeuralField._tau


def build_bcm_buffer_circuit():
    source = jp.CustomInput("source", (1, 1, 1))
    target = jp.CustomInput("target", (1, 1, 1))
    reward = jp.CustomInput("reward", (1,))
    source.set_data(np.ones((1, 1, 1), dtype=np.float32))
    target.set_data(np.ones((1, 1, 1), dtype=np.float32))
    reward.set_data(np.zeros((1,), dtype=np.float32))
    bcm = jp.BCMConnection(
        "bcm",
        (1, 1, 1),
        (1, 1, 1),
        tau_weights=1.0,
        tau_theta=1.0,
        learning_rate=0.1,
        min_theta=0.0,
        use_fixed_theta=True,
        fixed_theta=0.25,
        norm_target=0.0,
        norm_rate=0.0,
        safeguard_thr=-1.0,
    )
    source >> bcm
    target >> bcm.in1
    reward >> bcm.in2
    return bcm
