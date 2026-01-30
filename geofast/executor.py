"""
Executor: Dispatches functions to appropriate backends
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Callable
import time
import os

from .core import Backend, ProcessedFunction, get_config
from .utils import detect_backends, chunk_data, log


class GeoFastExecutor:
    """
    Executes ProcessedFunctions on the appropriate backend.
    Handles fallback, auto-selection, and hybrid dispatch.
    """
    
    def __init__(self):
        self.config = get_config()
        self.available_backends = detect_backends()
        self._gpu_executor = None
        self._numba_funcs = {}
    
    def run(self, processed_func: ProcessedFunction, *args, **kwargs) -> Any:
        """Execute a processed function on its designated backend"""
        backend = self._resolve_backend(processed_func, *args, **kwargs)
        
        log(f"Executing {processed_func.func.__name__} on {backend.name}")
        start = time.perf_counter()
        
        try:
            if backend == Backend.CPU:
                result = self._run_cpu(processed_func.func, *args, **kwargs)
            elif backend == Backend.CPU_PARALLEL:
                result = self._run_parallel(processed_func, *args, **kwargs)
            elif backend == Backend.GPU:
                result = self._run_gpu(processed_func.func, *args, **kwargs)
            elif backend == Backend.NUMBA:
                result = self._run_numba(processed_func.func, *args, **kwargs)
            elif backend == Backend.NUMBA_CUDA:
                result = self._run_numba_cuda(processed_func.func, *args, **kwargs)
            elif backend == Backend.HYBRID:
                result = self._run_hybrid(processed_func, *args, **kwargs)
            else:
                result = self._run_cpu(processed_func.func, *args, **kwargs)
            
            elapsed = time.perf_counter() - start
            log(f"Completed in {elapsed:.3f}s")
            return result
            
        except Exception as e:
            log(f"Error on {backend.name}: {e}")
            # Try fallback
            if backend != processed_func.fallback:
                log(f"Falling back to {processed_func.fallback.name}")
                return self.run(
                    processed_func.with_backend(processed_func.fallback),
                    *args, **kwargs
                )
            raise
    
    def _resolve_backend(
        self, 
        processed_func: ProcessedFunction, 
        *args, 
        **kwargs
    ) -> Backend:
        """Determine which backend to actually use"""
        requested = processed_func.backend
        
        # If not AUTO, check availability and return or use fallback
        if requested != Backend.AUTO:
            if self._is_available(requested):
                return requested
            log(f"{requested.name} not available, using {processed_func.fallback.name}")
            return processed_func.fallback
        
        # AUTO selection logic
        item_count = self._estimate_item_count(processed_func, *args, **kwargs)
        
        # Large datasets -> GPU if available
        if item_count >= self.config.auto_gpu_min_items:
            if self._is_available(Backend.GPU):
                return Backend.GPU
        
        # Medium datasets -> parallel CPU
        if item_count >= self.config.auto_parallel_min_items:
            return Backend.CPU_PARALLEL
        
        # Small datasets -> single CPU
        return Backend.CPU
    
    def _is_available(self, backend: Backend) -> bool:
        """Check if a backend is available"""
        if backend in (Backend.CPU, Backend.CPU_PARALLEL):
            return True
        if backend == Backend.GPU:
            return self.available_backends.get('cupy', False)
        if backend == Backend.NUMBA:
            return self.available_backends.get('numba', False)
        if backend == Backend.NUMBA_CUDA:
            return self.available_backends.get('numba_cuda', False)
        if backend == Backend.HYBRID:
            return self.available_backends.get('cupy', False)
        return True
    
    def _estimate_item_count(
        self, 
        processed_func: ProcessedFunction, 
        *args, 
        **kwargs
    ) -> int:
        """Estimate number of items to process for auto-selection"""
        # Check explicit item_count_arg
        if processed_func.item_count_arg:
            if processed_func.item_count_arg in kwargs:
                data = kwargs[processed_func.item_count_arg]
            else:
                # Try to find by position
                import inspect
                sig = inspect.signature(processed_func.func)
                params = list(sig.parameters.keys())
                if processed_func.item_count_arg in params:
                    idx = params.index(processed_func.item_count_arg)
                    if idx < len(args):
                        data = args[idx]
                    else:
                        return 0
                else:
                    return 0
            
            if hasattr(data, '__len__'):
                return len(data)
        
        # Heuristic: check first arg
        if args and hasattr(args[0], '__len__'):
            return len(args[0])
        
        return 0
    
    # =========================================================================
    # Backend implementations
    # =========================================================================
    
    def _run_cpu(self, func: Callable, *args, **kwargs) -> Any:
        """Single-threaded CPU execution"""
        return func(*args, **kwargs)
    
    def _run_parallel(
        self, 
        processed_func: ProcessedFunction, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Multi-core CPU execution.
        If batch=True, chunks the first argument and runs in parallel.
        """
        func = processed_func.func
        
        if not processed_func.batch:
            # Not a batch function, just run it (might be I/O bound)
            return func(*args, **kwargs)
        
        # Batch mode: chunk first argument
        if not args:
            return func(*args, **kwargs)
        
        data = args[0]
        rest_args = args[1:]
        
        if not hasattr(data, '__len__') or len(data) == 0:
            return func(*args, **kwargs)
        
        chunks = chunk_data(data, self.config.chunk_size)
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(func, chunk, *rest_args, **kwargs)
                for chunk in chunks
            ]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        # Flatten results
        return _flatten_results(results)
    
    def _run_gpu(self, func: Callable, *args, **kwargs) -> Any:
        """GPU execution via CuPy/cuSpatial"""
        try:
            import cupy as cp
            
            # Convert numpy arrays to cupy
            gpu_args = []
            for arg in args:
                if hasattr(arg, '__array__'):
                    gpu_args.append(cp.asarray(arg))
                else:
                    gpu_args.append(arg)
            
            gpu_kwargs = {}
            for k, v in kwargs.items():
                if hasattr(v, '__array__'):
                    gpu_kwargs[k] = cp.asarray(v)
                else:
                    gpu_kwargs[k] = v
            
            result = func(*gpu_args, **gpu_kwargs)
            
            # Convert back to numpy if needed
            if hasattr(result, 'get'):
                return result.get()
            return result
            
        except ImportError:
            raise RuntimeError("CuPy not available for GPU execution")
    
    def _run_numba(self, func: Callable, *args, **kwargs) -> Any:
        """JIT-compiled CPU execution via Numba"""
        try:
            from numba import njit
            
            # Cache JIT-compiled version
            func_id = id(func)
            if func_id not in self._numba_funcs:
                self._numba_funcs[func_id] = njit(func)
            
            jit_func = self._numba_funcs[func_id]
            return jit_func(*args, **kwargs)
            
        except ImportError:
            raise RuntimeError("Numba not available")
    
    def _run_numba_cuda(self, func: Callable, *args, **kwargs) -> Any:
        """CUDA kernel execution via Numba"""
        try:
            from numba import cuda
            import numpy as np
            
            # This assumes func is already a cuda.jit kernel
            # The user is responsible for proper kernel signature
            return func(*args, **kwargs)
            
        except ImportError:
            raise RuntimeError("Numba CUDA not available")
    
    def _run_hybrid(
        self, 
        processed_func: ProcessedFunction, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Split work between CPU and GPU.
        Uses ThreadPoolExecutor to run both simultaneously.
        """
        func = processed_func.func
        
        if not args or not hasattr(args[0], '__len__'):
            return self._run_cpu(func, *args, **kwargs)
        
        data = args[0]
        rest_args = args[1:]
        n = len(data)
        
        # Split data
        split_idx = int(n * self.config.cpu_gpu_split)
        cpu_data = data[:split_idx]
        gpu_data = data[split_idx:]
        
        results = [None, None]
        
        def run_cpu_portion():
            results[0] = self._run_parallel(
                processed_func.with_backend(Backend.CPU_PARALLEL),
                cpu_data, *rest_args, **kwargs
            )
        
        def run_gpu_portion():
            results[1] = self._run_gpu(func, gpu_data, *rest_args, **kwargs)
        
        # Run both in parallel using threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            cpu_future = executor.submit(run_cpu_portion)
            gpu_future = executor.submit(run_gpu_portion)
            cpu_future.result()
            gpu_future.result()
        
        # Combine results
        return _flatten_results(results)


def _flatten_results(results: List[Any]) -> Any:
    """Flatten list of results into single result"""
    if not results:
        return []
    
    # If results are lists, concatenate
    if isinstance(results[0], list):
        flat = []
        for r in results:
            flat.extend(r)
        return flat
    
    # If results are numpy arrays, concatenate
    try:
        import numpy as np
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results)
    except ImportError:
        pass
    
    # If results are dataframes, concatenate
    try:
        import pandas as pd
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
    except ImportError:
        pass
    
    # Otherwise return as list
    return results
