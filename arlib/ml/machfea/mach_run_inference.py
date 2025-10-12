from os import system,popen
import time
import os
import sys
import subprocess
# import datetime
from functools import partial
from multiprocessing import Pool
import json
import numpy as np
import random

seed = [0]

def RunSeed(data_index,seed):
    if data_index in test_set.keys():
        par2list = test_set[data_index]
    else:
        par2list = par2_dict['train'][data_index]
    if np.min(par2list) == 2400:
        return None
    target_file = "~/sibly/sibyl/data/Comp/non-incremental/" + str(data_index)
    if not os.path.exists(target_file):
        print("no instance ", data_index)
        return None
    file_name = data_index.replace('/','_')
    command = "python get_feature.py " + str(target_file) + " --dataset " + str(dataset)
    
    
    print(command)
    output = popen(command).read()
    tmp = output.split('\n')
    return tmp


if __name__ == '__main__':
    dataset = "ELA"
    with open("../data/SMTCompLabels.json",'r', encoding='UTF-8') as f:
        par2_dict = json.load(f)

    seed_ = int(sys.argv[1])

    
    dataset_name = 'Equality+LinearArith'

    ## For the json file whose format is like SMTCompLabels.json
    par2_dict = par2_dict[dataset_name]

    test_set = par2_dict['test']
    key_set = list(test_set.keys()) + list(par2_dict['train'].keys())
    print(len(key_set))

    p = Pool(processes=20)

    partial_RunSeed = partial(RunSeed,seed=seed_)
    ret = p.map(partial_RunSeed, key_set)
    
    # print(ret)

    ###save features
    fea_dict = {}
    
    for l in ret:
        if not l == None:
            fea_dict[l[-2]] = list(map(float,l[-3].replace('[','').replace(']','').replace(' ','').split(',')))

    with open('infer_result/' + str(dataset_name) + '_feature.json', 'w', newline='') as f:
        json.dump(fea_dict, f) 
    ###

    p.close()
    p.join()


            
            