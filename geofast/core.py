"""
Core components: Backend enum, config, and the @process decorator
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any, Union
from functools import wraps
import os


class Backend(Enum):
    """Available processing backends"""
    CPU = auto()          # Single-threaded CPU
    CPU_PARALLEL = auto() # Multi-core CPU via ProcessPoolExecutor
    GPU = auto()          # CUDA via CuPy/cuSpatial
    NUMBA = auto()        # JIT-compiled CPU
    NUMBA_CUDA = auto()   # Numba CUDA kernels
    HYBRID = auto()       # Split work between CPU and GPU
    AUTO = auto()         # Automatically select best backend


@dataclass
class GeoFastConfig:
    """Global configuration for GeoFast"""
    # CPU settings
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    chunk_size: int = 1000
    
    # GPU settings
    gpu_device: int = 0
    gpu_memory_limit: Optional[int] = None  # MB, None = no limit
    gpu_batch_size: int = 10000
    
    # Hybrid settings
    gpu_threshold: int = 5000  # Items above this go to GPU
    cpu_gpu_split: float = 0.3  # 30% to CPU, 70% to GPU in hybrid mode
    
    # Auto-selection thresholds
    auto_gpu_min_items: int = 1000
    auto_parallel_min_items: int = 100
    
    # Logging
    verbose: bool = False


# Global config instance
_config = GeoFastConfig()


def get_config() -> GeoFastConfig:
    return _config


def set_config(**kwargs):
    global _config
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise ValueError(f"Unknown config option: {key}")


@dataclass
class ProcessedFunction:
    """Wrapper that holds function metadata and dispatch logic"""
    func: Callable
    backend: Backend
    fallback: Backend
    batch: bool
    item_count_arg: Optional[str]
    
    def __call__(self, *args, **kwargs):
        from .executor import GeoFastExecutor
        executor = GeoFastExecutor()
        return executor.run(self, *args, **kwargs)
    
    def with_backend(self, backend: Backend) -> 'ProcessedFunction':
        """Override backend for this call"""
        return ProcessedFunction(
            func=self.func,
            backend=backend,
            fallback=self.fallback,
            batch=self.batch,
            item_count_arg=self.item_count_arg
        )
    
    # Allow direct access to underlying function
    def __getattr__(self, name):
        return getattr(self.func, name)


def process(
    backend: Union[Backend, str] = Backend.AUTO,
    fallback: Union[Backend, str] = Backend.CPU,
    batch: bool = False,
    item_count_arg: Optional[str] = None,
) -> Callable:
    """
    Decorator to mark a function for optimized processing.
    
    Args:
        backend: Which backend to use (or AUTO for automatic selection)
        fallback: Backend to use if primary isn't available
        batch: If True, function receives chunks of data
        item_count_arg: Argument name to check for auto-selecting backend
    
    Examples:
        @process(backend=Backend.GPU)
        def compute_distances(points_a, points_b):
            ...
        
        @process(backend=Backend.AUTO, item_count_arg="geometries")
        def simplify_geometries(geometries):
            ...
        
        @process(backend=Backend.CPU_PARALLEL, batch=True)
        def convert_files(filepaths):
            ...
    """
    # Handle string backend names
    if isinstance(backend, str):
        backend = Backend[backend.upper()]
    if isinstance(fallback, str):
        fallback = Backend[fallback.upper()]
    
    def decorator(func: Callable) -> ProcessedFunction:
        return ProcessedFunction(
            func=func,
            backend=backend,
            fallback=fallback,
            batch=batch,
            item_count_arg=item_count_arg,
        )
    
    return decorator


# Convenience decorators for common patterns
def cpu(func: Callable) -> ProcessedFunction:
    """Shorthand for @process(backend=Backend.CPU)"""
    return process(backend=Backend.CPU)(func)


def parallel(func: Callable) -> ProcessedFunction:
    """Shorthand for @process(backend=Backend.CPU_PARALLEL)"""
    return process(backend=Backend.CPU_PARALLEL)(func)


def gpu(func: Callable) -> ProcessedFunction:
    """Shorthand for @process(backend=Backend.GPU)"""
    return process(backend=Backend.GPU)(func)


def numba_jit(func: Callable) -> ProcessedFunction:
    """Shorthand for @process(backend=Backend.NUMBA)"""
    return process(backend=Backend.NUMBA)(func)


def hybrid(func: Callable) -> ProcessedFunction:
    """Shorthand for @process(backend=Backend.HYBRID)"""
    return process(backend=Backend.HYBRID)(func)


def auto(func: Callable) -> ProcessedFunction:
    """Shorthand for @process(backend=Backend.AUTO)"""
    return process(backend=Backend.AUTO)(func)
