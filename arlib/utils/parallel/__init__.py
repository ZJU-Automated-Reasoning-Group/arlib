"""Unified, concise parallel execution utilities and patterns."""

from .executor import ParallelExecutor, parallel_map, run_tasks
from .patterns import fork_join, pipeline, PipelineStage, producer_consumer, master_slave
from .actor import ActorSystem, spawn, ActorRef
from .stream import Stream
from .dataflow import Dataflow, Node

__all__ = [
    "ParallelExecutor",
    "parallel_map",
    "run_tasks",
    # patterns
    "fork_join",
    "pipeline",
    "PipelineStage",
    "producer_consumer",
    "master_slave",
    # actor
    "ActorSystem",
    "spawn",
    "ActorRef",
    # streaming
    "Stream",
    # dataflow
    "Dataflow",
    "Node",
]
