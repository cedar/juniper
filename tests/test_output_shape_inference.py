import pytest
import jax.numpy as jnp

import juniper as jp
from juniper.core.frontend import CircuitContext
from juniper.util import util
from juniper.util import util_jax


@pytest.fixture
def arch():
    architecture = jp.get_arch()
    architecture.clean()
    architecture.engine.clean()
    CircuitContext.set_current(architecture)
    yield architecture
    architecture.clean()
    architecture.engine.clean()
    CircuitContext.set_current(architecture)


def test_shuffle_image_infers_viewport_shape(arch):
    image = jp.CustomInput("image", (10, 12, 3))
    learn = jp.CustomInput("learn", (1,))
    shuffle = jp.ShuffleImage("shuffle", input_shape=(10, 12), viewport_size=(4, 4))

    image >> shuffle
    learn >> shuffle.learn_node
    arch.compile()

    assert shuffle.out0.shape == (4, 4, 3)
    assert shuffle.buffer_map["elapsed_learn_time"].shape == (1,)
    assert shuffle.buffer_map["learn_onset"].shape == (1,)


def test_space_to_rate_code_infers_scalar_output(arch):
    field = jp.CustomInput("field", (10,))
    rate = jp.SpaceToRateCode("rate", shape=(10,), limits=((0, 9),))

    field >> rate
    arch.compile()

    assert rate.out0.shape == ()
    assert rate.buffer_map["peak_pos"].shape == ()


def test_vector_to_scalars_preserves_trailing_input_shape(arch):
    vector = jp.CustomInput("vector", (2, 1))
    scalars = jp.VectorToScalars("scalars", N_scalars=2)

    vector >> scalars
    arch.compile()

    assert scalars.out0.shape == (1,)
    assert scalars.out1.shape == (1,)


def test_static_gain_returns_juniper_default_dtype(arch):
    gain = jp.StaticGain("gain", factor=1)

    output = gain.compute_kernel(
        {util.DEFAULT_INPUT_SLOT: jnp.array([1], dtype=jnp.uint8)},
        {},
    )

    assert output[util.DEFAULT_OUTPUT_SLOT].dtype == jnp.dtype(util_jax.cfg["jdtype"])


def test_reward_gated_hebbian_onset_buffer_is_boolean(arch):
    connection = jp.HebbianConnection(
        "connection",
        source_shape=(1,),
        target_shape=(1,),
        reward_type="reward_gated",
    )

    assert connection.buffer_map["reward_onset"].dtype == jnp.bool
