# coding: utf8
import os
import subprocess
from threading import Timer
import zlib
import logging
from typing import List, Optional, Union, Tuple
from pathlib import Path
import signal
import psutil
from dataclasses import dataclass
from enum import Enum
import time
import contextlib


class ColorScheme(Enum):
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class ProcessResult:
    """Container for process execution results"""
    output: str
    error: Optional[str]
    return_code: int
    execution_time: float
    memory_usage: float  # Peak memory usage in MB
    timeout_occurred: bool


class ColoredLogger:
    """Enhanced colored logging functionality"""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(console_handler)

    def _format_message(self, color: ColorScheme, *args) -> str:
        message = " ".join(str(arg) for arg in args)
        return f"{color.value}{message}{ColorScheme.END.value}"

    def info(self, *args) -> None:
        self.logger.info(self._format_message(ColorScheme.BLUE, *args))

    def warning(self, *args) -> None:
        self.logger.warning(self._format_message(ColorScheme.WARNING, *args))

    def error(self, *args) -> None:
        self.logger.error(self._format_message(ColorScheme.FAIL, *args))

    def debug(self, *args) -> None:
        self.logger.debug(self._format_message(ColorScheme.CYAN, *args))


class ProcessExecutor:
    """Handles process execution with timeout and resource monitoring"""

    def __init__(self, logger: Optional[ColoredLogger] = None):
        self.logger = logger or ColoredLogger(__name__)

    @staticmethod
    def get_process_memory(process: psutil.Process) -> float:
        """Get current memory usage of process in MB"""
        try:
            return process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    @contextlib.contextmanager
    def process_timeout(self, process: subprocess.Popen, timeout: float):
        """Context manager for process timeout handling"""
        timer = Timer(timeout, self._terminate_process, args=[process])
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def _terminate_process(self, process: subprocess.Popen) -> None:
        """Terminate process and its children"""
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
        except psutil.NoSuchProcess:
            pass

    def execute_with_timeout(self,
                             cmd: List[str],
                             timeout: float = 300,
                             cwd: Optional[str] = None) -> ProcessResult:
        """Execute command with timeout and resource monitoring"""
        start_time = time.time()
        max_memory = 0.0
        timeout_occurred = False

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                preexec_fn=os.setsid
            )

            with self.process_timeout(process, timeout):
                while process.poll() is None:
                    try:
                        proc = psutil.Process(process.pid)
                        current_memory = self.get_process_memory(proc)
                        max_memory = max(max_memory, current_memory)
                        time.sleep(0.1)
                    except psutil.NoSuchProcess:
                        break

                stdout, stderr = process.communicate()

            return_code = process.returncode
            execution_time = time.time() - start_time

            return ProcessResult(
                output=stdout.decode('utf-8', errors='replace'),
                error=stderr.decode('utf-8', errors='replace') if stderr else None,
                return_code=return_code,
                execution_time=execution_time,
                memory_usage=max_memory,
                timeout_occurred=timeout_occurred
            )

        except subprocess.TimeoutExpired:
            self._terminate_process(process)
            return ProcessResult(
                output="",
                error="Process timed out",
                return_code=-1,
                execution_time=timeout,
                memory_usage=max_memory,
                timeout_occurred=True
            )


class FileUtils:
    """File handling utilities"""

    @staticmethod
    def is_executable(fpath: Union[str, Path]) -> bool:
        """Check if file exists and is executable"""
        if not fpath:
            return False
        path = Path(fpath)
        return path.is_file() and os.access(path, os.X_OK)

    @staticmethod
    def which(program: Union[str, List[str]]) -> Optional[str]:
        """Find executable in PATH"""
        programs = [program] if isinstance(program, str) else program

        for prog in programs:
            path = Path(prog)
            if path.is_absolute() and FileUtils.is_executable(path):
                return str(path)

            for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                exe_path = Path(path_dir) / prog
                if FileUtils.is_executable(exe_path):
                    return str(exe_path)
        return None

    @staticmethod
    def compress_file(file_path: Union[str, Path]) -> bytes:
        """Compress file content using zlib"""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content = path.read_text(encoding='utf-8')
            return zlib.compress(content.encode('utf-8'))
        except Exception as e:
            raise IOError(f"Error compressing file {file_path}: {e}")

    @staticmethod
    def find_files(
            directory: Union[str, Path],
            pattern: str = "*.smt2",
            recursive: bool = True
    ) -> List[Path]:
        """Find files matching pattern in directory"""
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        glob_pattern = "**/" + pattern if recursive else pattern
        return list(path.glob(glob_pattern))


# Example usage
if __name__ == "__main__":
    logger = ColoredLogger("TestLogger")
    executor = ProcessExecutor(logger)
    file_utils = FileUtils()

    # Test file search
    try:
        smt2_files = file_utils.find_files("./test_dir", "*.smt2")
        logger.info(f"Found {len(smt2_files)} SMT2 files")
    except Exception as e:
        logger.error(f"Error finding files: {e}")

    # Test process execution
    cmd = ["echo", "hello world"]
    result = executor.execute_with_timeout(cmd, timeout=5)
    logger.info(f"Process output: {result.output.strip()}")
    logger.info(f"Execution time: {result.execution_time:.2f}s")
    logger.info(f"Memory usage: {result.memory_usage:.2f}MB")
