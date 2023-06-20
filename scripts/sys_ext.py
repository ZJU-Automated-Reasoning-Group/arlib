# coding: utf8
import os
import subprocess
from threading import Timer
import zlib


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColoredPrint:

    @staticmethod
    def info(*args):
        msg = " ".join(["{}".format(i) for i in args])
        print(bcolors.OKBLUE + msg + bcolors.ENDC)
        # print(bcolors.OKGREEN + msg + bcolors.ENDC)

    @staticmethod
    def warning(*args):
        msg = " ".join(["{}".format(i) for i in args])
        print(bcolors.WARNING + msg + bcolors.ENDC)

    @staticmethod
    def error(*args):
        msg = " ".join(["{}".format(i) for i in args])
        print(bcolors.FAIL + msg + bcolors.ENDC)


def is_exec(fpath):
    if fpath is None:
        return False
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    if isinstance(program, str):
        choices = [program]
    else:
        choices = program

    for p in choices:
        fpath, _ = os.path.split(p)
        if fpath:
            if is_exec(p):
                return p
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, p)
                if is_exec(exe_file):
                    return exe_file
    return None


def get_file_size(fname):
    return os.path.getsize(fname)


def string_compress(fpath):
    str_content = open(fpath).read()
    zlib_str = zlib.compress(str_content)
    return zlib_str


def terminate(process, is_timeout):
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception:
            # print("error for interrupting")
            # print(ex)
            pass


def solve_with_bin_solver(cmd, timeout=300):
    # ret = "unknown"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    is_timeout = [False]
    timer = Timer(timeout, terminate, args=[p, is_timeout])
    timer.start()
    out = p.stdout.readlines()
    out = ' '.join([str(element.decode('UTF-8')) for element in out])
    p.stdout.close()
    timer.cancel()
    if p.poll() is None:
        p.terminate()
    if is_timeout[0]:
        return "timeout"
    return out


def find_smt2_files(path):
    flist = []  # path to smtlib2 files
    for root, dirs, files in os.walk(path):
        for fname in files:
            if os.path.splitext(fname)[1] == '.smt2':
                flist.append(os.path.join(root, fname))
    return flist
