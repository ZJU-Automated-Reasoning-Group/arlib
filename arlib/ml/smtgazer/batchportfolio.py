from os import system
import time
import os
import sys
import subprocess
# import datetime
from functools import partial
from multiprocessing import Pool
from collections import deque
import json
import numpy as np

from os import popen
seed = [0,1,2,3,4,5,6,7,8,9]

def RunSeed(seed,key):
    c_num = 20
    if key == "Equality+LinearArith":
        c_num=2
    if key == "SyGuS":
        c_num=3
    command = ""
    ### Training
    # command = "python -u SMTportfolio.py train -dataset " + str(key) + " -solverdict machfea/" + str(key) + "_solver.json -seed " + str(seed) + " -cluster_num "+str(c_num)
    ### Testing
    # command = "python -u SMTportfolio.py infer -clusterPortfolio output/train_result_" + str(key) + "_4_"+str(c_num)+"_" + str(seed) + ".json -dataset " + str(key) + " -solverdict machfea/" + str(key) + "_solver.json -seed " + str(seed)
    print(command)
    output = popen(command).read()
    print(output)

if __name__ == '__main__':
    key_set = ['Equality+LinearArith','QF_NonLinearIntArith',"QF_Bitvec","SyGuS","BMC","SymEx"]
    p = Pool(processes=10)
    for key in key_set:
        print(key)
        partial_RunSeed = partial(RunSeed,key = key)
        p.map(partial_RunSeed, seed)

    p.close()
    p.join()
