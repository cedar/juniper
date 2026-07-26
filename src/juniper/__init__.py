# Expose Juniper api
# api
from .core.frontend.Circuit import Circuit
from .core.backend.Simulation import SimulationRuntime
from .core.backend.Simulation import close_connections
from .core.backend.Simulation import init_prng
from .core.backend.Simulation import load_buffers
from .core.backend.Simulation import open_connections
from .core.backend.Simulation import refresh_prng
from .core.backend.Simulation import reset_state
from .core.backend.Simulation import run_simulation
from .core.backend.Simulation import save_buffers
from .core.backend.Simulation import trace

# architecture
from .core.Architecture import delete_arch
from .core.Architecture import get_arch
from .core.Architecture import init_logging
from .core.Architecture import init_logging_to_file

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
from .arrays.ReorderAxes import ReorderAxes
from .arrays.Resize import Resize
from .arrays.Projection import Projection
from .arrays.VectorToScalars import VectorToScalars
from .arrays.ScalarsToVector import ScalarsToVector

# configurable math classes
from .math.Gaussian import Gaussian
from .math.LateralKernel import LateralKernel

# dft
from .dft.HebbianConnection import HebbianConnection
from .dft.NeuralField import NeuralField
from .dft.SpaceToRateCode import SpaceToRateCode
from .dft.RateToSpaceCode import RateToSpaceCode
from .dft.BCMConnection import BCMConnection

# image_processing
from .image_processing.ColorConversion import ColorConversion
from .image_processing.DNN import DNN
from .image_processing.ColorFMap import ColorFMap
from .image_processing.ViewportCamera import ViewportCamera
from .image_processing.ShuffleImage import ShuffleImage
from .image_processing.RemoveBlackWhiteGreys import RemoveBlackWhiteGreys
from .image_processing.RGB2HSV import RGB2HSV

# sinks
from .sinks.TCPWriter import TCPWriter
from .sinks.StaticDebug import StaticDebug

# sources
from .sources.CustomInput import CustomInput
from .sources.DemoInput import DemoInput
from .sources.GaussInput import GaussInput
from .sources.ImageLoader import ImageLoader
from .sources.TCPReader import TCPReader
from .sources.TimedBoost import TimedBoost

# robotics
from . import robotics

# loggingimport logging
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ =[
    "SimulationRuntime",
    "trace",
    "init_prng",
    "refresh_prng",
    "load_buffers",
    "save_buffers",
    "reset_state",
    "run_simulation",
    "close_connections",
    "open_connections",
    "Circuit",
    "AddConstant",
    "ComponentMultiply",
    "Convolution",
    "Normalization",
    "StaticGain",
    "Sum",
    "TransferFunction",
    "Clamp",
    "CompressAxes",
    "ExpandAxes",
    "Flip",
    "MatrixPadding",
    "MatrixSlice",
    "ReorderAxes",
    "Resize",
    "Projection",
    "Gaussian",
    "LateralKernel",
    "HebbianConnection",
    "BCMConnection",
    "NeuralField",
    "SpaceToRateCode",
    "RateToSpaceCode",
    "DNN",
    "ColorConversion",
    "ColorFMap",
    "ViewportCamera",
    "ShuffleImage",
    "RemoveBlackWhiteGreys",
    "RGB2HSV",
    "TCPWriter",
    "StaticDebug",
    "CustomInput",
    "DemoInput",
    "GaussInput",
    "ImageLoader",
    "TCPReader",
    "TimedBoost",
    "robotics",
    "get_arch",
    "delete_arch",
    "init_logging",
    "init_logging_to_file",
    "VectorToScalars",
    "ScalarsToVector"
]
