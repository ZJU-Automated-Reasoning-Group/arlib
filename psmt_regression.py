"""
Run benchmarks in the dir named "benchmarks" (Currently, from the seeds of SemanticFusion)
"""
from ctypes import util
import os
from re import RegexFlag
import subprocess

from arlib.utils.types import SolverResult
import psmt_main

regression = []

for root_out, dirs_out, files_out in os.walk('./benchmarks'):
    for dir in filter(lambda x: 'QF' in x, dirs_out):
        print(f'Enter {dir}', flush=True)
        for root, _, files in os.walk(os.path.join(root_out, dir, 'sat')):
            print(f'Enter {root}', flush=True)
            for file in files:
                print(f'Test {file}', flush=True)
                result = psmt_main.process_file(os.path.join(root, file), 'ALL')
                if SolverResult.SAT != result:
                    print(f'Inconsistent {file}: expect SAT but got {result}', flush=True)
                    regression.append(file)
        for root, _, files in os.walk(os.path.join(root_out, dir, 'unsat')):
            print(f'Enter {root}', flush=True)
            for file in files:
                print(f'Test {file}', flush=True)
                result = psmt_main.process_file(os.path.join(root, file), 'ALL')
                if SolverResult.UNSAT != result:
                    print(f'Inconsistent {file}: expect UNSAT but got {result}', flush=True)
                    regression.append(file)

print(f'--------\n{regression}\n')
