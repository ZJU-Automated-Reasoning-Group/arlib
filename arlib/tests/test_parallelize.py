import os
import sys
import time
from unittest.mock import Mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arlib.parallel.parallelize import (
    Mutex, ProcessScheduler,
    run_tasks, serial_tasks_from_commands, Task, TaskGroup, TaskState)


class TestTaskGroup:
    def test_empty(self):
        empty = TaskGroup([])
        assert empty.empty()

    def test_non_empty(self):
        non_empty = TaskGroup([Task(["process"])])
        assert not non_empty.empty()

    def test_empty_after_popping_single_task(self):
        g = TaskGroup([Task(["process"])])
        task = g.pop_next()
        assert task
        assert g.empty()

    def test_two_parallel_tasks(self):
        g = TaskGroup([Task(["process1"]),
                       Task(["process2"])])
        first_task = g.pop_next()
        assert first_task
        second_task = g.pop_next()
        assert second_task
        assert g.empty()

    def test_next_exclusive_task_is_available_only_after_first_one_is_done(self):
        mutex = Mutex()
        g = TaskGroup([Task(["process1"], mutex=mutex),
                       Task(["process2"], mutex=mutex)])
        first_task = g.pop_next()
        assert first_task
        next_task = g.pop_next()
        assert next_task is None
        assert not g.empty()

        first_task.done(exit_code=0)
        next_task = g.pop_next()
        assert next_task
        assert g.empty()

    def test_next_dependent_task_is_available_only_after_dependency_is_done(self):
        dependency = Task(["process1"])
        dependant = Task(["process2"], dependencies=[dependency])
        g = TaskGroup([dependency, dependant])
        assert g.pop_next() == dependency
        assert not g.empty()
        assert g.pop_next() is None
        assert not g.empty()

        dependency.done(exit_code=0)
        assert g.pop_next() == dependant
        assert g.empty()

    def test_dependent_task_fails_if_dependency_failed(self):
        dependency = Task(["process1"])
        dependant = Task(["process2"], dependencies=[dependency])
        g = TaskGroup([dependency, dependant])
        assert g.pop_next() == dependency
        dependency.done(exit_code=1)
        assert dependency.state() == TaskState.FAILED
        assert g.pop_next() is None
        assert g.empty()
        assert dependant.state() == TaskState.FAILED


class TestRunTasks:
    def test_serial_tasks_do_not_run_in_parallel(self):
        scheduler = ProcessScheduler(3)
        handle_exit_code = Mock()
        tasks = serial_tasks_from_commands([_one_second_cmd()] * 3,
                                           handle_exit_code=handle_exit_code)
        start = time.time()
        run_tasks(tasks, scheduler)

        # all three 1-second tasks should be run sequentially and take 3 seconds
        stop = time.time()
        assert stop - start < 3.7
        assert stop - start >= 3
        handle_exit_code.assert_called_with(0)
        assert handle_exit_code.call_count == 3

    def test_tasks_run_in_parallel(self):
        scheduler = ProcessScheduler(3)
        handle_exit_code = Mock()
        tasks = [Task(cmd, handle_exit_code) for cmd in [_one_second_cmd()] * 3]
        start = time.time()
        run_tasks(tasks, scheduler)

        # all three 1-second tasks should be run in parallel
        # and take about 1 second to finish
        stop = time.time()
        assert stop - start < 1.5
        assert stop - start >= 1
        handle_exit_code.assert_called_with(0)
        assert handle_exit_code.call_count == 3

    def test_exclusive_and_non_exclusive_tasks_mixed(self):
        scheduler = ProcessScheduler(3)
        handle_exit_code = Mock()
        mutex = Mutex()
        exclusive_tasks = [Task(cmd, handle_exit_code, mutex=mutex)
                           for cmd in [_one_second_cmd()] * 2]
        parallel_tasks = [Task(cmd, handle_exit_code)
                          for cmd in [_one_second_cmd()] * 4]
        start = time.time()
        run_tasks(exclusive_tasks + parallel_tasks, scheduler)

        # while 2 exclusive tasks should take at least 2 seconds
        # to run sequentially, other 4 tasks should take at least 2 seconds
        # to run in parallel using 2 processes left
        stop = time.time()
        assert stop - start < 2.5
        assert stop - start >= 2
        handle_exit_code.assert_called_with(0)
        assert handle_exit_code.call_count == 6


def test():
    exit_codes = []

    def check_exit_code(expected_exit_code):
        def check(exit_code):
            assert exit_code == expected_exit_code
            exit_codes.append(exit_code)

        return check

    scheduler = ProcessScheduler(3)
    proc_num = 10
    for i in range(proc_num):
        scheduler.add(Task(_one_second_cmd(i), check_exit_code(i)))
    scheduler.wait()
    assert len(exit_codes) == proc_num


def _one_second_cmd(exit_code: int = 0):
    return [sys.executable, __file__, str(exit_code)]


def print_hello():
    print("hello")


def simple_test():
    tt = TestTaskGroup()
    tt.test_two_parallel_tasks()


if __name__ == "__main__":
    # code = int(sys.argv[1])
    code = 0
    print(f"{code}: stdout", flush=True)
    print(f"{code}: stderr", flush=True, file=sys.stderr)
    time.sleep(1)
    simple_test()
    sys.exit(code)
