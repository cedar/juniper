import functools
import json
import os
import traceback
import juniper as jp
import numpy as np
import pytest
import time
from contextlib import contextmanager
import io
import sys

from juniper.util import util_jax


def function_test(func):
    """Decorator to clean arch after every test."""
    functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            raise RuntimeError(f"test_function {func.__name__} failed with the above stderr call")
        finally:
            arch = jp.get_arch()
            arch.clean()
            arch.engine.clean()
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

class TestJuniper:
    """
    A simple test class for pytesting. The class owns the architecture singleton, 
    while each test function defines a small test circuit tailord to a specific test case.
    """
    arch = jp.get_arch(name="test_arch")


    @function_test
    def test_recording(self):
        in1 = jp.CustomInput("in1", {"shape":(1,)})
        in1.set_data(np.ones((1,)))

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        recording, timing = self.arch.run_simulation(num_steps=42, steps_to_record=["in1"], print_timing=False, save_buffer=False)
        assert len(recording) == 42
        assert np.isclose(recording[0][0], 1)
        assert np.isclose(np.sum(recording), 42)

    @function_test
    def test_custom_input_updates_after_compile(self):
        """Sources should push new CPU-side data into runtime state each tick."""
        in1 = jp.CustomInput("in1", {"shape":(2,)})
        in1.set_data(np.array([1, 2], dtype=np.float32))

        self.arch.compile(warmup=1, print_compile_info=False, load_buffer=False)
        recording1, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["in1"], print_timing=False)

        in1.set_data(np.array([3, 4], dtype=np.float32))
        recording2, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["in1"], print_timing=False)

        assert np.allclose(recording1[-1][0], np.array([1, 2], dtype=np.float32))
        assert np.allclose(recording2[-1][0], np.array([3, 4], dtype=np.float32))


    @function_test
    def test_static_arch(self):
        """test: 1*3+1=4"""
        const = jp.AddConstant("constant", {"constant":2})
        in1 = jp.GaussInput("input", {"shape":(1,), "sigma":(0.01,), "center": (0.,), "amplitude":1})
        in2 = jp.GaussInput("input2", {"shape":(1,), "sigma":(0.01,), "center": (0.,), "amplitude":1})
        mult = jp.ComponentMultiply("mult", {})
        out = jp.Sum("out", {})
        in1 >> const >> mult >> out
        in2 >> out

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        recording, timing = self.arch.run_simulation(num_steps=3, steps_to_record=["out"], print_timing=False, save_buffer=False)

        last_out = recording[-1]
        res = np.sum(last_out)

        assert np.isclose(res, 4)

    @function_test
    def test_juniper_syntax(self):
        """test all syntactic categories implemented in the juniper dsl"""
        const = jp.AddConstant("constant", {"constant":2})
        in1 = jp.GaussInput("input", {"shape":(1,), "sigma":(0.01,), "center": (0.,), "amplitude":1})
        in2 = jp.GaussInput("input2", {"shape":(1,), "sigma":(0.01,), "center": (0.,), "amplitude":1})
        mult = jp.ComponentMultiply("mult", {})
        out = jp.Sum("out", {})
        out2 = jp.Sum("out2", {})
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

        last_out = recording[-1][0]
        last_out2 = recording[-1][1]

        res = np.sum(last_out)
        res2 = np.sum(last_out2)

        assert np.isclose(res, 4)
        assert np.isclose(res2, 7)

    @function_test
    def test_nested_circuit(self):
        """Tests if nested circuits are handled correctly."""
        circ = jp.Circuit("circ", {})
        with circ as c:
            summed_inputs = jp.Sum("summed_inputs", {})
            c.set_input("in0", summed_inputs)
            c.set_input("in1", summed_inputs)
            c.set_input("in2", summed_inputs)
            c.set_input("in3", summed_inputs)
            c.set_output("out0", summed_inputs)

        circ2 = jp.Circuit("circ2", {})
        with circ2 as c:
            summed_inputs = jp.Sum("summed_inputs", {})
            c.set_input("in0", summed_inputs)
            c.set_output("out0", summed_inputs)

        
        external_input = jp.CustomInput("external_input", {"shape":(1,)})
        external_input.set_data(np.ones((1,)))

        external_input >> circ.in0
        external_input >> circ.in1
        external_input >> circ.in2
        external_input >> circ.in3

        circ >> circ2

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        self.arch.reset_state()
        recording, timing = self.arch.run_simulation(num_steps=1, steps_to_record=["circ", "circ2"], print_timing=False, save_buffer=False)

        last_out = recording[-1][0]
        last_out2 = recording[-1][1]

        res = np.sum(last_out)
        res2 = np.sum(last_out2)

        assert np.isclose(res, 4)
        assert np.isclose(res2, 4)

    @function_test
    def test_compile_info_flattened_nested_circuit(self):
        """Top-level CompileInfo should expose nested elements with extended paths."""
        circ = jp.Circuit("circ", {})
        with circ as c:
            summed_inputs = jp.Sum("summed_inputs", {})
            c.set_input("in0", summed_inputs)
            c.set_output("out0", summed_inputs)

        external_input = jp.CustomInput("external_input", {"shape":(1,)})
        external_input.set_data(np.ones((1,)))
        external_input >> circ

        self.arch.compile(warmup=0, print_compile_info=False, load_buffer=False)

        compile_info = self.arch.engine.compile_info
        assert ("circ",) in compile_info.compiled_elements
        assert ("circ", "summed_inputs") in compile_info.compiled_elements
        assert compile_info.compiled_elements[("circ", "summed_inputs")].element.get_name() == "summed_inputs"
        assert "circ" in compile_info.kernel_map
        assert "summed_inputs" in compile_info.kernel_map["circ"]["sub_kernel"]

    @function_test
    def test_component_multiply_product_aggregation(self):
        """ComponentMultiply should multiply multiple incoming values instead of summing them."""
        in1 = jp.CustomInput("in1", {"shape":(1,)})
        in2 = jp.CustomInput("in2", {"shape":(1,)})
        in3 = jp.CustomInput("in3", {"shape":(1,)})
        in1.set_data(np.array([2], dtype=np.float32))
        in2.set_data(np.array([3], dtype=np.float32))
        in3.set_data(np.array([4], dtype=np.float32))
        mult = jp.ComponentMultiply("mult", {})

        in1 >> mult
        in2 >> mult
        in3 >> mult

        self.arch.compile(warmup=1, print_compile_info=False, load_buffer=False)
        recording, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["mult"], print_timing=False)

        assert np.isclose(np.sum(recording[-1][0]), 24)

    @function_test
    def test_duplicate_element_name_fails(self):
        """Circuit construction should reject duplicate element names."""
        jp.CustomInput("in1", {"shape":(1,)})
        with pytest.raises(Exception):
            jp.CustomInput("in1", {"shape":(1,)})

    @function_test
    def test_max_incoming_connection_fails(self):
        """Slots with one allowed incoming connection should reject a second source."""
        in1 = jp.CustomInput("in1", {"shape":(1,)})
        in2 = jp.CustomInput("in2", {"shape":(1,)})
        out = jp.AddConstant("out", {"constant": 1})
        in1 >> out
        with pytest.raises(Exception):
            in2 >> out

    @function_test
    def test_reset(self):
        """tests if reset correctly restores the initial state and doesn't cause retracing from jax."""
        nf1 = jp.NeuralField("nf1", {"shape": (50,50), "sigmoid": "ExpSigmoid", "beta": 100, "theta": 0, "resting_level":-5, "global_inhibition":0, "input_noise_gain":0, "tau":0.02})
        gauss = jp.GaussInput("gauss", {"sigma": (3,3), "shape":(50,50), "amplitude": 6, "center": (25,25)})
        gauss >> nf1

        self.arch.compile(warmup=0, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled

        t_comp = time.time()
        compile_recording, _ = self.arch.run_simulation(num_steps=1, steps_to_record=["nf1"], print_timing=False, save_buffer=False)
        t_comp = time.time() - t_comp
        compile_recording, _ = self.arch.run_simulation(num_steps=5, steps_to_record=["nf1"], print_timing=False, save_buffer=False)
        cache_size = getattr(self.arch.engine.tick, "_cache_size", None)
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

        res_nf_1 = np.sum(recording1[-1][0])
        res_nf_2 = np.sum(recording2[-1][0])
        res_nf_3 = np.sum(recording3[-1][0])

        assert recording1[-1][0].shape == (50, 50)
        assert recording3[-1][0].shape == (50, 50)
        assert recording1[-1][0].dtype == util_jax.cfg["dtype"]
        assert recording3[-1][0].dtype == util_jax.cfg["dtype"]
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
                dnn2 = jp.DNN("dnn2", {"layer":"4_3"})

    @function_test
    def test_buffer_save_and_load(self):
        """Permanent buffers should be saved to disk and loaded into a fresh state."""
        self.arch.set_arch_name("test_buffer_arch")
        data_file = util_jax.cfg["arch_file_path"] + self.arch.get_name() + ".data"
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

            self.arch.clean()
            self.arch.engine.clean()

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
        tcp_reader = jp.TCPReader("tcp_reader", tcp_params)
        tcp_writer = jp.TCPWriter("tcp_writer", tcp_params)

        input = jp.CustomInput("input", {"shape":(3,)})
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
                out_array = np.asanyarray(recording[-1][0])
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
        out = jp.Sum("out", {})
        in1 = jp.CustomInput("in1", {"shape":(2,)})
        in2 = jp.CustomInput("in2", {"shape":(2,2)})
        in1 >> out
        in2 >> out
        with pytest.raises(Exception):
            self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        self.arch.clean()

        out = jp.NeuralField("out", {"shape":(50,50),
                                     "sigmoid": "AbsSigmoid",
                                     "beta": 100,
                                     "theta":0,
                                     "resting_level":-5,
                                     "global_inhibition":0,
                                     "input_noise_gain":0,
                                     "tau":0.02})
        in1 = jp.CustomInput("in1", {"shape":(50,50)})
        in1 >> out
        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        self.arch.clean()

        # case 2
        out = jp.NeuralField("out", {"shape":(50,50),
                                     "sigmoid": "AbsSigmoid",
                                     "beta": 100,
                                     "theta":0,
                                     "resting_level":-5,
                                     "global_inhibition":0,
                                     "input_noise_gain":0,
                                     "tau":0.02})
        in1 = jp.CustomInput("in1", {"shape":(50,50)})
        in2 = jp.CustomInput("in2", {"shape":(1,)})
        proj = jp.CompressAxes("proj", {"axis": (0,1), "compression_type":"Maximum"})
        in1 >> out
        in2 >> out
        out >> proj >> out
        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        self.arch.clean()

        # case 3
        out = jp.NeuralField("out", {"shape":(50,50),
                                     "sigmoid": "AbsSigmoid",
                                     "beta": 100,
                                     "theta":0,
                                     "resting_level":-5,
                                     "global_inhibition":0,
                                     "input_noise_gain":0,
                                     "tau":0.02})
        in1 = jp.CustomInput("in1", {"shape":(50,50)})
        in2 = jp.CustomInput("in2", {"shape":(1,)})
        proj = jp.CompressAxes("proj", {"axis": (0,), "compression_type":"Maximum"})
        in1 >> out
        in2 >> out
        out >> proj >> out
        with pytest.raises(Exception):
            self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        self.arch.clean()

        # case 4
        out = jp.NeuralField("out", {"shape":(50,50),
                                     "sigmoid": "AbsSigmoid",
                                     "beta": 100,
                                     "theta":0,
                                     "resting_level":-5,
                                     "global_inhibition":0,
                                     "input_noise_gain":0,
                                     "tau":0.02})
        in1 = jp.CustomInput("in1", {"shape":(50,50)})
        in2 = jp.CustomInput("in2", {"shape":(1,)})
        proj = jp.CompressAxes("proj", {"axis": (0,1), "compression_type":"Maximum", "compress_all":True})
        in1 >> out
        in2 >> out
        out >> proj >> out
        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        self.arch.clean()

        # case 5
        out = jp.NeuralField("out", {"shape":(50,50),
                                     "sigmoid": "AbsSigmoid",
                                     "beta": 100,
                                     "theta":0,
                                     "resting_level":-5,
                                     "global_inhibition":0,
                                     "input_noise_gain":0,
                                     "tau":0.02})
        in1 = jp.CustomInput("in1", {"shape":(50,50)})
        in2 = jp.CustomInput("in2", {"shape":()})
        proj = jp.CompressAxes("proj", {"axis": (0,1), "compression_type":"Maximum", "compress_all":True})
        in1 >> out
        in2 >> out
        out >> proj >> out
        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        self.arch.clean()



        
    @function_test
    def test_nested_state_update(self):
        """Tests if steps in nested circuits update their state correctly and can be recorded."""
        c = jp.Circuit("c", {})
        with c as c:
            s = jp.Sum("s", {})
            c.set_input("in0", s)
        
        input = jp.CustomInput("input", {"shape":(3,)})
        in_array = np.asanyarray([1,2,3], dtype=np.float32)
        input.set_data(in_array)

        input >> c

        self.arch.compile(warmup=3, print_compile_info=False, load_buffer=False)
        assert self.arch.is_compiled
        
        recording, timing = self.arch.run_simulation(num_steps=1, steps_to_record=["c.s", "c.s.out0"], print_timing=False, save_buffer=False)

        out_array = np.asanyarray(recording[-1][0])
        explicit_out_array = np.asanyarray(recording[-1][1])

        assert np.array_equal(in_array, out_array)
        assert np.array_equal(in_array, explicit_out_array)
        





def build_bcm_buffer_circuit():
    source = jp.CustomInput("source", {"shape":(1, 1, 1)})
    target = jp.CustomInput("target", {"shape":(1, 1, 1)})
    reward = jp.CustomInput("reward", {"shape":(1,)})
    source.set_data(np.ones((1, 1, 1), dtype=np.float32))
    target.set_data(np.ones((1, 1, 1), dtype=np.float32))
    reward.set_data(np.zeros((1,), dtype=np.float32))
    bcm = jp.BCMConnection(
        "bcm",
        {
            "shape": (1, 1, 1),
            "target_shape": (1, 1, 1),
            "tau_weights": 1.0,
            "tau_theta": 1.0,
            "learning_rate": 0.1,
            "min_theta": 0.0,
            "use_fixed_theta": True,
            "fixed_theta": 0.25,
            "norm_target": 0.0,
            "norm_rate": 0.0,
            "safeguard_thr": -1.0,
        },
    )
    source >> bcm
    target >> bcm.in1
    reward >> bcm.in2
    return bcm
