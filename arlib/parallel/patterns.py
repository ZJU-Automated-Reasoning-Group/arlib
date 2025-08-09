"""Common parallel algorithm patterns built on top of ParallelExecutor.

Patterns:
- fork_join: submit independent tasks and join on completion
- pipeline: stage-based processing using bounded queues
- producer_consumer: one or more producers feeding a consumer pool
- master_slave: master creates tasks, slaves execute and return results
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Any

from .executor import ParallelExecutor

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")


def fork_join(tasks: Sequence[Callable[[], R]], *, kind: str = "threads", max_workers: Optional[int] = None) -> List[R]:
    """Run independent callables in parallel and join their results in order."""
    with ParallelExecutor(kind=kind, max_workers=max_workers) as ex:
        futures = [ex.submit(task) for task in tasks]
        return [f.result() for f in futures]


@dataclass
class PipelineStage:
    worker: Callable[[Any], Any]
    parallelism: int = 1


def pipeline(
    stages: Sequence[PipelineStage],
    inputs: Iterable[Any],
    *,
    queue_size: int = 64,
    kind: str = "threads",
) -> List[Any]:
    """Simple M-stage pipeline with bounded queues between stages.

    Each stage has its own ParallelExecutor with specified parallelism.
    """
    logger = logging.getLogger("arlib.parallel.pipeline")

    # Create stage queues
    qs = [queue.Queue(maxsize=queue_size) for _ in range(len(stages) + 1)]

    # Feed initial input
    def feeder():
        for item in inputs:
            qs[0].put(item)
        # Signal completion
        for _ in range(stages[0].parallelism):
            qs[0].put(StopIteration)

    threads: List[threading.Thread] = []
    threads.append(threading.Thread(target=feeder, daemon=True))
    threads[-1].start()

    # Stage workers
    for idx, stage in enumerate(stages):
        out_q = qs[idx + 1]
        in_q = qs[idx]

        def stage_worker():
            with ParallelExecutor(kind=kind, max_workers=stage.parallelism) as ex:
                while True:
                    item = in_q.get()
                    if item is StopIteration:
                        # propagate completion to next stage
                        for _ in range(stage.parallelism if idx + 1 < len(stages) else 1):
                            out_q.put(StopIteration)
                        break
                    res = ex.run(stage.worker, [item])
                    # res is a single-item list
                    out_q.put(res[0])

        t = threading.Thread(target=stage_worker, daemon=True)
        threads.append(t)
        t.start()

    # Collect results from final queue into a list
    results: List[Any] = []
    end_signals = 0
    while True:
        item = qs[-1].get()
        if item is StopIteration:
            end_signals += 1
            # Wait until all signals received from last stage
            if end_signals >= stages[-1].parallelism:
                break
            continue
        results.append(item)

    for t in threads:
        t.join(timeout=0.1)

    return results


def producer_consumer(
    produce: Callable[[queue.Queue], None],
    consume: Callable[[Any], R],
    *,
    consumer_parallelism: int = 1,
    kind: str = "threads",
    queue_size: int = 256,
) -> List[R]:
    """Run a producer feeding a pool of consumers; return results collected in order of completion."""
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    results: List[R] = []
    results_lock = threading.Lock()

    def consumer_loop():
        with ParallelExecutor(kind=kind, max_workers=consumer_parallelism) as ex:
            while True:
                item = q.get()
                if item is StopIteration:
                    break
                # Run single item
                out = ex.run(consume, [item])[0]
                with results_lock:
                    results.append(out)

    t_prod = threading.Thread(target=produce, args=(q,), daemon=True)
    t_cons = threading.Thread(target=consumer_loop, daemon=True)
    t_prod.start()
    t_cons.start()

    t_prod.join()
    # signal consumers to finish
    q.put(StopIteration)
    t_cons.join()
    return results


def master_slave(
    tasks: Iterable[Any],
    worker: Callable[[Any], R],
    *,
    max_workers: Optional[int] = None,
    kind: str = "threads",
) -> List[R]:
    """Master generates tasks; slaves execute and return results in input order."""
    with ParallelExecutor(kind=kind, max_workers=max_workers) as ex:
        return ex.run(worker, list(tasks))
