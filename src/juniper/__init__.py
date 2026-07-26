# Expose Juniper api
# api
# loggingimport logging
import logging

# robotics
from . import robotics

# algebra
from .algebra.AddConstant import AddConstant
from .algebra.ComponentMultiply import ComponentMultiply
from .algebra.Convolution import Convolution
from .algebra.Normalization import Normalization
from .algebra.StaticGain import StaticGain
from .algebra.Sum import Sum
from .algebra.TransferFunction import TransferFunction

# arrays
from .arrays.Clamp import Clamp
from .arrays. CompressAxes import CompressAxes
from .arrays.ExpandAxes import ExpandAxes
from .arrays.Flip import Flip
from .arrays.MatrixPadding import MatrixPadding
from .arrays.MatrixSlice import MatrixSlice
from .arrays.Projection import Projection
from .arrays.ReorderAxes import ReorderAxes
from .arrays.Resize import Resize
from .arrays.ScalarsToVector import ScalarsToVector
from .arrays.VectorToScalars import VectorToScalars

# architecture
from .core.Architecture import delete_arch, get_arch, init_logging, init_logging_to_file
from .core.backend.Simulation import (
    SimulationRuntime,
    close_connections,
    init_prng,
    load_buffers,
    open_connections,
    refresh_prng,
    reset_state,
    run_simulation,
    save_buffers,
    trace,
)
from .core.frontend.Circuit import Circuit
from .dft.BCMConnection import BCMConnection

# dft
from .dft.HebbianConnection import HebbianConnection
from .dft.NeuralField import NeuralField
from .dft.RateToSpaceCode import RateToSpaceCode
from .dft.SpaceToRateCode import SpaceToRateCode

# image_processing
from .image_processing.ColorConversion import ColorConversion
from .image_processing.ColorFMap import ColorFMap
from .image_processing.DNN import DNN
from .image_processing.RemoveBlackWhiteGreys import RemoveBlackWhiteGreys
from .image_processing.RGB2HSV import RGB2HSV
from .image_processing.ShuffleImage import ShuffleImage
from .image_processing.ViewportCamera import ViewportCamera

# configurable math classes
from .math.Gaussian import Gaussian
from .math.LateralKernel import LateralKernel
from .sinks.StaticDebug import StaticDebug

# sinks
from .sinks.TCPWriter import TCPWriter

# sources
from .sources.CustomInput import CustomInput
from .sources.DemoInput import DemoInput
from .sources.GaussInput import GaussInput
from .sources.ImageLoader import ImageLoader
from .sources.TCPReader import TCPReader
from .sources.TimedBoost import TimedBoost

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ =[
    "DNN",
    "RGB2HSV",
    "AddConstant",
    "BCMConnection",
    "Circuit",
    "Clamp",
    "ColorConversion",
    "ColorFMap",
    "ComponentMultiply",
    "CompressAxes",
    "Convolution",
    "CustomInput",
    "DemoInput",
    "ExpandAxes",
    "Flip",
    "GaussInput",
    "Gaussian",
    "HebbianConnection",
    "ImageLoader",
    "LateralKernel",
    "MatrixPadding",
    "MatrixSlice",
    "NeuralField",
    "Normalization",
    "Projection",
    "RateToSpaceCode",
    "RemoveBlackWhiteGreys",
    "ReorderAxes",
    "Resize",
    "ScalarsToVector",
    "ShuffleImage",
    "SimulationRuntime",
    "SpaceToRateCode",
    "StaticDebug",
    "StaticGain",
    "Sum",
    "TCPReader",
    "TCPWriter",
    "TimedBoost",
    "TransferFunction",
    "VectorToScalars",
    "ViewportCamera",
    "close_connections",
    "delete_arch",
    "get_arch",
    "init_logging",
    "init_logging_to_file",
    "init_prng",
    "load_buffers",
    "open_connections",
    "refresh_prng",
    "reset_state",
    "robotics",
    "run_simulation",
    "save_buffers",
    "trace"
]
