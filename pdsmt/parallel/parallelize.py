"""
Taken from https://gitlab.com/GrammaTech/lifter-eval/-/blob/main/bin/parallelize.py
"""
import shlex
import shutil
import subprocess
import sys
import tempfile
from enum import Enum
from typing import (
    IO, Any, Callable, Iterable, List, Optional, Sequence, Tuple, cast)


def format_cmd(cmd):
    return " ".join(map(shlex.quote, cmd))


def print_cmd(cmd):
    print(format_cmd(cmd), flush=True)


class Mutex:
    def __init__(self):
        self.__locked = False

    def try_lock(self):
        if self.__locked:
            return False
        self.__locked = True
        return True

    def release(self):
        assert self.__locked
        self.__locked = False


class TaskState(Enum):
    PENDING = 0
    EXECUTING = 1
    FAILED = 2
    SUCCEEDED = 3


class Task:
    def __init__(self,
                 cmd: Sequence[str],
                 handle_exit_code: Optional[Callable[[int], None]] = None,
                 mutex: Optional[Mutex] = None,
                 dependencies: List["Task"] = None,
                 description: Optional[str] = None
                 ):
        self.cmd = cmd
        self.__handle_exit_code = handle_exit_code
        self.__mutex = mutex
        self.__dependencies = dependencies or []
        self.description = description
        self.__state = TaskState.PENDING

    def state(self):
        return self.__state

    def try_execute(self):
        if self.__state != TaskState.PENDING:
            return
        for d in self.__dependencies:
            if d.__state in [TaskState.PENDING, TaskState.EXECUTING]:
                return
            if d.__state == TaskState.FAILED:
                self.__state = TaskState.FAILED
                return
        if self.__mutex and not self.__mutex.try_lock():
            return

        self.__state = TaskState.EXECUTING

    def done(self, exit_code: int):
        assert self.__state == TaskState.EXECUTING

        if self.__mutex:
            self.__mutex.release()

        if exit_code == 0:
            self.__state = TaskState.SUCCEEDED
        else:
            self.__state = TaskState.FAILED

        if self.__handle_exit_code:
            self.__handle_exit_code(exit_code)


class ProcessScheduler:
    def __init__(self, max):
        self.__max = max
        self.__processes: List[Tuple[subprocess.Popen, Any, Task]] = []
        self.__stdout_captured = False

    def add(self, task: Task):
        if len(self.__processes) == self.__max:
            self.wait_for_any_process()

        stdout = None
        if self.__stdout_captured:
            stdout = tempfile.TemporaryFile()
        if task.description:
            print(task.description)
        print(f"starting: " + format_cmd(task.cmd), flush=True)
        proc = subprocess.Popen(
            task.cmd, stdin=subprocess.PIPE, stdout=stdout, stderr=stdout)
        _log_process(proc.pid, "started")
        cast(IO[bytes], proc.stdin).close()
        if stdout is None:
            self.__stdout_captured = True
        self.__processes.append((proc, stdout, task))

    def wait(self):
        while len(self.__processes):
            self.wait_for_any_process()

    def wait_for_any_process(self):
        while True:
            i = 0
            while i < len(self.__processes):
                p, stdout, task = self.__processes[i]
                exit_code = p.poll()
                if exit_code is None and i == len(self.__processes) - 1:
                    # wait a little bit avoiding CPU consumption
                    try:
                        exit_code = p.wait(0.1)
                    except subprocess.TimeoutExpired:
                        pass

                if exit_code is None:
                    i += 1
                else:
                    del self.__processes[i]
                    if stdout is None:
                        self.__stdout_captured = False
                    else:
                        _move_to_stdout(stdout)
                    _log_process(p.pid, f"exited with {exit_code}")
                    task.done(exit_code)
                    return


def _log_process(pid: int, description: str):
    print(f"PID #{pid}: " + description, flush=True)


def _move_to_stdout(fo):
    try:
        fo.seek(0)
        shutil.copyfileobj(fo, sys.stdout.buffer)
        sys.stdout.flush()
    finally:
        fo.close()


class TaskGroup:
    def __init__(self, tasks: Iterable[Task]):
        self.__tasks = list(tasks)

    def tasks(self):
        return self.__tasks

    def empty(self):
        return len(self.__tasks) == 0

    def pop_next(self) -> Optional[Task]:
        i = 0
        while i < len(self.__tasks):
            t = self.__tasks[i]
            t.try_execute()
            if t.state() != TaskState.PENDING:
                del self.__tasks[i]
                if t.state() == TaskState.EXECUTING:
                    return t
            else:
                i += 1
        return None


def serial_tasks_from_commands(
        cmds: Iterable[Sequence[str]],
        handle_exit_code: Optional[Callable[[int], None]] = None,
        description: Optional[str] = None):
    result = []
    dependency: Optional[Task] = None
    for cmd in cmds:
        dependencies: List[Task] = [dependency] if dependency else []
        description = description if dependency is None else None
        task = Task(
            cmd,
            handle_exit_code=handle_exit_code,
            dependencies=dependencies,
            description=description)
        result.append(task)
        dependency = task
    return result


def run_tasks(tasks: Iterable[Task], scheduler: ProcessScheduler):
    g = TaskGroup(tasks)
    while not g.empty():
        task = g.pop_next()
        if task:
            scheduler.add(task)
        else:
            scheduler.wait_for_any_process()
    scheduler.wait()
