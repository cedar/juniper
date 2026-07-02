import pytest

import juniper as jp
from juniper.core.backend.Compiler import Compiler
from juniper.core.backend.Exceptions import CompilerError
from juniper.core.frontend import CircuitContext
from juniper.core.frontend.Step import Step


class BrokenOutputStep(Step):
    """Step used to exercise compiler diagnostics for missing output specs."""

    def __init__(self, name):
        super().__init__(name, {}, [], is_dynamic=False)
        self.compute_kernel = lambda inputs, state, **kwargs: state

    def infer_output_shapes(self, input_specs):
        return {}


class BrokenBufferStep(Step):
    """Step used to exercise diagnostics for missing buffer specs."""

    def __init__(self, name):
        super().__init__(name, {}, [], is_dynamic=False)
        self.register_buffer("state", None)
        self.compute_kernel = lambda inputs, state, **kwargs: state


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


def test_compile_failure_report_traces_all_branches(arch):
    source = jp.CustomInput("source", (1,))
    broken = BrokenOutputStep("broken")
    left = jp.Sum("left")
    right = jp.Sum("right")

    source >> broken
    broken >> left
    broken >> right

    with pytest.raises(CompilerError) as error:
        Compiler.compile(arch)

    report = str(error.value)
    assert "BrokenOutputStep('broken') [OUTPUT_SLOT_DTYPE_UNRESOLVED:out0, OUTPUT_SLOT_SHAPE_UNRESOLVED:out0]" in report
    assert "BrokenOutputStep('broken') -> Sum('left')" in report
    assert "BrokenOutputStep('broken') -> Sum('right')" in report
    assert "UPSTREAM_ELEMENT_UNCOMPILED" in report


def test_compile_failure_report_identifies_dependency_cycles(arch):
    first = BrokenOutputStep("first")
    second = BrokenOutputStep("second")

    first >> second
    second >> first

    with pytest.raises(CompilerError) as error:
        Compiler.compile(arch)

    report = str(error.value)
    assert "CYCLIC_FAILURE_DEPENDENCY" in report
    assert "Failure dependency cycles:" in report
    assert "BrokenOutputStep('first') -> BrokenOutputStep('second') -> BrokenOutputStep('first')" in report


def test_compile_failure_report_traces_multiple_upstream_sources(arch):
    first_input = jp.CustomInput("first_input", (1,))
    second_input = jp.CustomInput("second_input", (1,))
    first = BrokenOutputStep("first")
    second = BrokenOutputStep("second")
    joined = jp.Sum("joined")

    first_input >> first >> joined
    second_input >> second >> joined

    with pytest.raises(CompilerError) as error:
        Compiler.compile(arch)

    report = str(error.value)
    assert "BrokenOutputStep('first') -> Sum('joined')" in report
    assert "BrokenOutputStep('second') -> Sum('joined')" in report


def test_compile_failure_report_identifies_unresolved_buffers(arch):
    source = jp.CustomInput("source", (1,))
    broken = BrokenBufferStep("broken")
    source >> broken

    with pytest.raises(CompilerError) as error:
        Compiler.compile(arch)

    report = str(error.value)
    assert "BrokenBufferStep('broken') [BUFFER_SHAPE_UNRESOLVED:state]" in report
