"""
Felix Enterprise Suite

Hardware-aware deep learning optimization platform for enterprise deployment.
"""

__version__ = "1.0.0"
__author__ = "Felix Technologies Inc."
__email__ = "support@felix-enterprise.com"

# Core components
from .client import Client
from .config import Settings
from .exceptions import FelixEnterpriseError, OptimizationError, HardwareError

# API models
from .models import (
    OptimizationJob,
    JobStatus,
    HardwareTarget,
    OptimizationConfig,
    PerformanceMetrics,
)

# Hardware integrations
from .hardware import (
    AMDROCmBackend,
    CUDABackend,
    IntelGPUBackend,
    get_available_backends,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Core classes
    "Client",
    "Settings",
    
    # Exceptions
    "FelixEnterpriseError",
    "OptimizationError", 
    "HardwareError",
    
    # Models
    "OptimizationJob",
    "JobStatus",
    "HardwareTarget",
    "OptimizationConfig",
    "PerformanceMetrics",
    
    # Hardware backends
    "AMDROCmBackend",
    "CUDABackend", 
    "IntelGPUBackend",
    "get_available_backends",
]