"""Lightweight parallel execution helpers.

Features:
- Unified process/thread pool via a single class
- Submit, map, and gather results
- Optional task-level logging (start/end/duration)
- Graceful shutdown and timeouts
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, Any, Dict


T = TypeVar("T")
R = TypeVar("R")


PoolKind = Union[ThreadPoolExecutor, ProcessPoolExecutor]


@dataclass
class ParallelExecutor:
    """Unified wrapper around thread/process pools.

    kind: "threads" or "processes".
    """

    max_workers: Optional[int] = None
    kind: str = "threads"  # "threads" | "processes"
    initializer: Optional[Callable[..., None]] = None
    initargs: Optional[tuple] = None
    log_events: bool = False
    logger: Optional[logging.Logger] = None
    cancel_on_error: bool = True
    kill_pool_on_timeout: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("threads", "processes"):
            raise ValueError("kind must be 'threads' or 'processes'")
        if self.logger is None:
            self.logger = logging.getLogger("arlib.parallel")
        if self.kind == "threads":
            self._pool: PoolKind = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers, initializer=self.initializer, initargs=self.initargs
            )

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any):
        if self.log_events:
            # Use a top-level wrapper to remain picklable for processes
            return self._pool.submit(
                _execute_with_logging, fn, args, kwargs, self.logger.name if self.logger else None
            )
        return self._pool.submit(fn, *args, **kwargs)

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
        return list(self._pool.map(fn, items))

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def run(
        self,
        fn: Callable[[T], R],
        items: Iterable[T],
        timeout: Optional[float] = None,
        *,
        return_exceptions: bool = False,
    ) -> List[R]:
        futures = [self.submit(fn, item) for item in items]
        return _gather_results(
            futures,
            timeout=timeout,
            logger=self.logger,
            cancel_on_error=self.cancel_on_error,
            kill_pool_on_timeout=self.kill_pool_on_timeout,
            return_exceptions=return_exceptions,
        )

    # Optional context manager convenience
    def __enter__(self) -> "ParallelExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    kind: str = "threads",
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Optional[tuple] = None,
) -> List[R]:
    """Convenience parallel map over a sequence.

    Example:
        results = parallel_map(work, data, kind="processes", max_workers=8)
    """
    ex = ParallelExecutor(
        max_workers=max_workers, kind=kind, initializer=initializer, initargs=initargs
    )
    try:
        return ex.run(fn, items, timeout=timeout)
    finally:
        ex.shutdown()


def run_tasks(
    tasks: Sequence[Tuple[Callable[..., R], tuple, dict]],
    *,
    kind: str = "threads",
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Optional[tuple] = None,
) -> List[R]:
    """Run heterogeneous callables with different args/kwargs.

    tasks: list of (callable, args, kwargs)
    """
    ex = ParallelExecutor(
        max_workers=max_workers, kind=kind, initializer=initializer, initargs=initargs
    )
    try:
        futures = [ex.submit(fn, *args, **kwargs) for fn, args, kwargs in tasks]
        return _gather_results(futures, timeout=timeout, logger=ex.logger,
                               cancel_on_error=ex.cancel_on_error,
                               kill_pool_on_timeout=ex.kill_pool_on_timeout,
                               return_exceptions=False)
    finally:
        ex.shutdown()


def _execute_with_logging(
    fn: Callable[..., R], args: tuple, kwargs: dict, logger_name: Optional[str]
) -> R:
    """Execute a callable, logging start/end/duration using a named logger.

    This function is top-level so it's picklable for process pools.
    """
    logger = logging.getLogger(logger_name) if logger_name else None
    name = getattr(fn, "__name__", repr(fn))
    start = time.time()
    if logger:
        logger.debug("task.start name=%s", name)
    try:
        return fn(*args, **kwargs)
    finally:
        if logger:
            logger.debug("task.end name=%s elapsed=%.6fs", name, time.time() - start)


def _gather_results(
    futures: Sequence,
    *,
    timeout: Optional[float],
    logger: Optional[logging.Logger],
    cancel_on_error: bool,
    kill_pool_on_timeout: bool,
    return_exceptions: bool,
) -> List:
    """Collect results with robust handling for exceptions and timeouts.

    - Preserves submission order
    - On exception: optionally cancels remaining futures and re-raises (or returns exceptions)
    - On timeout: cancels remaining futures; optionally signals to kill pool by raising TimeoutError
    """
    results: List[Any] = []
    try:
        for f in futures:
            try:
                results.append(f.result(timeout=timeout))
            except Exception as exc:  # includes TimeoutError
                if logger:
                    logger.error("task.error type=%s msg=%s", type(exc).__name__, exc)
                if isinstance(exc, TimeoutError):
                    # best-effort cancel remaining
                    for rem in futures:
                        if not rem.done():
                            rem.cancel()
                    if kill_pool_on_timeout:
                        # Let caller decide to recreate pool; propagate timeout
                        raise
                    if return_exceptions:
                        results.append(exc)
                        continue
                    raise
                # Non-timeout exceptions
                if return_exceptions:
                    results.append(exc)
                    continue
                if cancel_on_error:
                    for rem in futures:
                        if not rem.done():
                            rem.cancel()
                raise
        return results
    finally:
        # nothing to close here; shutdown is handled by caller/owner of the pool
        pass
