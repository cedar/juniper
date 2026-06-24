# Expose Juniper api
# api
from .core.frontend.Circuit import Circuit
from .core.backend.Engine import Engine

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

# error types
from .core.backend.Exceptions import JuniperError
from .core.backend.Exceptions import CompilerError
from .core.backend.Exceptions import ShapeInferenceError
from .core.backend.Exceptions import TypeInferenceError
from .core.backend.Exceptions import LoadBufferError
from .core.backend.Exceptions import SaveBufferError
from .core.backend.Exceptions import EngineError
from .core.backend.Exceptions import NotCompiledError
from .core.backend.Exceptions import CircuitError
from .core.backend.Exceptions import CircuitConnectionError
from .core.backend.Exceptions import TCPError
from .core.backend.Exceptions import RecordingError
from .core.backend.Exceptions import JuniperConfigurationError
from .core.backend.Exceptions import JuniperUserError

# warnings
from .core.backend.Warnings import JuniperWarning
from .core.backend.Warnings import CompilerWarning
from .core.backend.Warnings import ShapeInferenceWarning
from .core.backend.Warnings import TypeInferenceWarning
from .core.backend.Warnings import LoadBufferWarning
from .core.backend.Warnings import SaveBufferWarning
from .core.backend.Warnings import EngineWarning
from .core.backend.Warnings import NotCompiledWarning
from .core.backend.Warnings import CircuitWarning
from .core.backend.Warnings import CircuitConnectionWarning
from .core.backend.Warnings import TCPWarning
from .core.backend.Warnings import RecordingWarning
from .core.backend.Warnings import JuniperConfigurationWarning
from .core.backend.Warnings import JuniperUserWarning

# loggingimport logging
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ =[
    "Engine",
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
    "ScalarsToVector",
    "JuniperError",
    "CompilerError",
    "ShapeInferenceError",
    "TypeInferenceError",
    "LoadBufferError",
    "SaveBufferError",
    "EngineError",
    "NotCompiledError",
    "CircuitError",
    "CircuitConnectionError",
    "TCPError",
    "RecordingError",
    "JuniperConfigurationError",
    "JuniperUserError",
    "JuniperWarning",
    "CompilerWarning",
    "ShapeInferenceWarning",
    "TypeInferenceWarning",
    "LoadBufferWarning",
    "SaveBufferWarning",
    "EngineWarning",
    "NotCompiledWarning",
    "CircuitWarning",
    "CircuitConnectionWarning",
    "TCPWarning",
    "RecordingWarning",
    "JuniperConfigurationWarning",
    "JuniperUserWarning"
]
