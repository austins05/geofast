"""
High-performance parallel processing utilities.
Uses shared memory to avoid serialization overhead.
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import shared_memory, Pool, cpu_count
from typing import Callable, List, Any, Tuple, Optional, Iterator
import pickle
from functools import partial

from .core import get_config


# =============================================================================
# Shared Memory Utilities
# =============================================================================

class SharedNumpyArray:
    """
    Numpy array backed by shared memory for zero-copy parallel access.

    Example:
        >>> arr = np.random.rand(1000000)
        >>> shared = SharedNumpyArray.from_array(arr)
        >>> # Pass shared.name to workers
        >>> result = shared.to_numpy()
        >>> shared.close()
    """

    def __init__(self, name: str, shape: Tuple, dtype: np.dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self._shm = None
        self._arr = None

    @classmethod
    def from_array(cls, arr: np.ndarray, name: Optional[str] = None) -> 'SharedNumpyArray':
        """Create shared memory from existing array."""
        if name is None:
            name = f"geofast_{id(arr)}_{os.getpid()}"

        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shared_arr[:] = arr[:]

        obj = cls(name, arr.shape, arr.dtype)
        obj._shm = shm
        obj._arr = shared_arr
        return obj

    @classmethod
    def attach(cls, name: str, shape: Tuple, dtype: np.dtype) -> 'SharedNumpyArray':
        """Attach to existing shared memory (for workers)."""
        obj = cls(name, shape, dtype)
        obj._shm = shared_memory.SharedMemory(name=name)
        obj._arr = np.ndarray(shape, dtype=dtype, buffer=obj._shm.buf)
        return obj

    def to_numpy(self) -> np.ndarray:
        """Get the numpy array view."""
        if self._arr is None:
            self._shm = shared_memory.SharedMemory(name=self.name)
            self._arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self._shm.buf)
        return self._arr

    def close(self):
        """Close shared memory (call from main process when done)."""
        if self._shm is not None:
            self._shm.close()

    def unlink(self):
        """Unlink shared memory (removes it completely)."""
        if self._shm is not None:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


class SharedArrayContext:
    """
    Context manager for shared memory arrays.

    Example:
        >>> with SharedArrayContext() as ctx:
        ...     ctx.add('points', points_array)
        ...     ctx.add('polygons', poly_array)
        ...     results = parallel_map(worker_func, tasks, ctx.info)
    """

    def __init__(self):
        self.arrays = {}
        self.info = {}

    def add(self, name: str, arr: np.ndarray):
        """Add an array to shared memory."""
        shared = SharedNumpyArray.from_array(arr, name=f"geofast_{name}_{os.getpid()}")
        self.arrays[name] = shared
        self.info[name] = {
            'shm_name': shared.name,
            'shape': shared.shape,
            'dtype': shared.dtype
        }

    def get(self, name: str) -> np.ndarray:
        """Get array by name."""
        return self.arrays[name].to_numpy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for shared in self.arrays.values():
            shared.close()
            shared.unlink()


def get_shared_array(info: dict, name: str) -> np.ndarray:
    """
    Helper for workers to attach to shared array.

    Args:
        info: Dict from SharedArrayContext.info
        name: Array name

    Returns:
        Numpy array view of shared memory
    """
    arr_info = info[name]
    shm = shared_memory.SharedMemory(name=arr_info['shm_name'])
    return np.ndarray(arr_info['shape'], dtype=arr_info['dtype'], buffer=shm.buf)


# =============================================================================
# Parallel Processing Functions
# =============================================================================

def parallel_map(func: Callable, items: List[Any],
                 max_workers: Optional[int] = None,
                 chunksize: int = 1) -> List[Any]:
    """
    Parallel map using ProcessPoolExecutor.

    Args:
        func: Function to apply to each item (must be picklable)
        items: List of items to process
        max_workers: Number of workers (default: cpu_count)
        chunksize: Items per worker batch

    Returns:
        List of results in same order as items
    """
    if max_workers is None:
        max_workers = get_config().max_workers

    if len(items) == 0:
        return []

    if len(items) <= chunksize or max_workers == 1:
        return [func(item) for item in items]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items, chunksize=chunksize))

    return results


def parallel_starmap(func: Callable, args_list: List[Tuple],
                     max_workers: Optional[int] = None) -> List[Any]:
    """
    Parallel starmap - unpack arguments for each call.

    Args:
        func: Function to call with unpacked args
        args_list: List of argument tuples

    Returns:
        List of results
    """
    if max_workers is None:
        max_workers = get_config().max_workers

    if len(args_list) == 0:
        return []

    # Use Pool.starmap for proper argument unpacking
    with Pool(max_workers) as pool:
        results = pool.starmap(func, args_list)

    return results


def parallel_chunks(func: Callable, data: np.ndarray,
                    chunk_size: int = 10000,
                    max_workers: Optional[int] = None,
                    reduce_func: Optional[Callable] = None) -> Any:
    """
    Process array in parallel chunks.

    Args:
        func: Function to apply to each chunk
        data: Array to process
        chunk_size: Size of each chunk
        max_workers: Number of workers
        reduce_func: Optional function to combine results

    Returns:
        Combined results
    """
    if max_workers is None:
        max_workers = get_config().max_workers

    n = len(data)
    n_chunks = (n + chunk_size - 1) // chunk_size

    chunks = [data[i*chunk_size:min((i+1)*chunk_size, n)]
              for i in range(n_chunks)]

    results = parallel_map(func, chunks, max_workers=max_workers)

    if reduce_func is not None:
        return reduce_func(results)

    return results


def parallel_apply_shared(func: Callable, indices: List[int],
                          shared_info: dict,
                          max_workers: Optional[int] = None) -> List[Any]:
    """
    Apply function using shared memory arrays.

    Args:
        func: Function(index, shared_info) -> result
        indices: List of indices to process
        shared_info: Dict from SharedArrayContext.info
        max_workers: Number of workers

    Returns:
        List of results
    """
    if max_workers is None:
        max_workers = get_config().max_workers

    # Create partial function with shared_info bound
    worker = partial(func, shared_info=shared_info)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, indices))

    return results


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def batch_iterator(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Yield batches of items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def process_in_batches(func: Callable, items: List[Any],
                       batch_size: int = 1000,
                       progress_callback: Optional[Callable] = None) -> List[Any]:
    """
    Process items in batches with optional progress callback.

    Args:
        func: Function to process a batch of items
        items: All items to process
        batch_size: Items per batch
        progress_callback: Optional callback(completed, total)

    Returns:
        Flattened list of results
    """
    results = []
    total = len(items)
    completed = 0

    for batch in batch_iterator(items, batch_size):
        batch_results = func(batch)
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)

        completed += len(batch)
        if progress_callback:
            progress_callback(completed, total)

    return results


# =============================================================================
# Thread-based parallelism for I/O bound tasks
# =============================================================================

def parallel_map_threads(func: Callable, items: List[Any],
                         max_workers: Optional[int] = None) -> List[Any]:
    """
    Thread-based parallel map for I/O bound tasks.
    Lower overhead than processes, but shares GIL.

    Args:
        func: Function to apply
        items: Items to process
        max_workers: Number of threads

    Returns:
        List of results in order
    """
    if max_workers is None:
        max_workers = min(32, get_config().max_workers * 2)

    if len(items) == 0:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))

    return results


def parallel_io(func: Callable, items: List[Any],
                max_workers: int = 32) -> List[Any]:
    """
    Parallel I/O operations using thread pool.
    Optimized for file reads, network requests, etc.

    Args:
        func: I/O function to apply
        items: Items to process
        max_workers: Number of threads (default 32)

    Returns:
        List of results
    """
    return parallel_map_threads(func, items, max_workers=max_workers)


# =============================================================================
# Result Aggregation
# =============================================================================

def reduce_lists(results: List[List]) -> List:
    """Flatten list of lists."""
    flat = []
    for r in results:
        flat.extend(r)
    return flat


def reduce_sets(results: List[set]) -> set:
    """Union of sets."""
    combined = set()
    for r in results:
        combined.update(r)
    return combined


def reduce_dicts(results: List[dict], combine_func: Callable = None) -> dict:
    """
    Merge dictionaries.

    Args:
        results: List of dicts
        combine_func: Optional function to combine values for same key

    Returns:
        Merged dictionary
    """
    combined = {}
    for r in results:
        for k, v in r.items():
            if k in combined and combine_func:
                combined[k] = combine_func(combined[k], v)
            else:
                combined[k] = v
    return combined


def reduce_arrays(results: List[np.ndarray], axis: int = 0) -> np.ndarray:
    """Concatenate numpy arrays."""
    return np.concatenate(results, axis=axis)
