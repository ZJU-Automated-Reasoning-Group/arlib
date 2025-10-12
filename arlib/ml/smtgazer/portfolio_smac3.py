from ConfigSpace import Configuration, ConfigurationSpace, Float, Categorical

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario

from os import system
import time
import os
import sys
from collections import deque
import json
from copy import deepcopy


import numpy as np

x1 = 0
x2 = 0
x3 = 0
s1 = -1
s2 = -1
s3 = -1
s4 = -1
dataset = ""
seed = ""
w1 = 0.5
w2 = 0.5
start_idx = 0
for i in range(len(sys.argv)-1):
    if (sys.argv[i] == '-t1'):
        x1 = float(sys.argv[i+1])
    elif(sys.argv[i] == '-t2'):
        x2 = float(sys.argv[i+1])
    elif(sys.argv[i] == '-t3'):
        x3 = float(sys.argv[i+1])
    elif(sys.argv[i] == '-s1'):
        s1 = int(sys.argv[i+1])
    elif(sys.argv[i] == '-s2'):
        s2 = int(sys.argv[i+1])
    elif(sys.argv[i] == '-s3'):
        s3 = int(sys.argv[i+1])
    elif(sys.argv[i] == '-s4'):
        s4 = int(sys.argv[i+1])
    elif(sys.argv[i] == '-cluster'):
        cluster = int(sys.argv[i+1])

    elif(sys.argv[i] == '-dataset'):
        dataset = str(sys.argv[i+1])
    elif(sys.argv[i] == '-seed'):
        seed = str(sys.argv[i+1])
    elif(sys.argv[i] == '-w1'):
        w1 = float(sys.argv[i+1])
    elif(sys.argv[i] == '-si'):
        start_idx = int(sys.argv[i+1])

x4 = 1200-x1-x2-x3
w2 = 1-w1
def train(config: Configuration, seed: int = 0) -> float:

    dataset=config['dataset']
    cluster=config['cluster']
    import random
    SEED = 1
    random.seed(SEED)
    if dataset == "ELA":
        dataset = "Equality+LinearArith"

    if dataset == "Equality+LinearArith":
        dataplace = "ELA"
    elif dataset == "QF_Bitvec":
        dataplace = "QFBV"
    elif dataset == "QF_NonLinearIntArith":
        dataplace = "QFNIA"
    else:
        dataplace = dataset
    tc = "tmp/machfea_infer_result_" + str(dataset) + "_train_feature_train_" + config['seed'] + ".json"
    td = "data/" + str(dataset) + "Labels.json"

    with open(td,'r', encoding='UTF-8') as f:
        par2_dict = json.load(f)
    with open(tc,'r', encoding='UTF-8') as f:
        train_cluster_dict = json.load(f)

    train_set = par2_dict['train']
    print(len(train_set))

    time = [2400 for i in range(len(train_set))]

    output_idx = []

    if config['s1'] != -1:
        output_idx.append(config['s1'])
    if config['s2'] != -1:
        output_idx.append(config['s2'])
    if config['s3'] != -1:
        output_idx.append(config['s3'])
    if config['s4'] != -1:
        output_idx.append(config['s4'])

    time = 0
    fail = 0

    tmp = [config['t1'],config['t2'],config['t3'],config['t4']]
    all_ = 0
    for i in range(len(output_idx)):
        all_ += tmp[i]
    final_config = [config['t1']/all_*1200,config['t2']/all_*1200,config['t3']/all_*1200,config['t4']/all_*1200]

    key_set = list(train_set.keys())
    full_key_set = list(train_set.keys())
    key_set = []
    for j in full_key_set:
        if j in train_cluster_dict.keys() and train_cluster_dict[j] == cluster:
            key_set.append(j)
        if  "./infer_result/"+str(dataset)+"/_data_sibly_sibyl_data_"+str(dataset)+"_"+str(dataset)+"_" + j.replace("/","_") + ".json" in train_cluster_dict.keys() and train_cluster_dict["./infer_result/"+str(dataset)+"/_data_sibly_sibyl_data_"+str(dataset)+"_"+str(dataset)+"_" + j.replace("/","_") + ".json"] == cluster:
            key_set.append(j)
        if  "./infer_result/"+str(dataplace)+"/_data_sibly_sibyl_data_Comp_non-incremental_" + j.replace("/","_") + ".json" in train_cluster_dict.keys() and train_cluster_dict["./infer_result/"+str(dataplace)+"/_data_sibly_sibyl_data_Comp_non-incremental_" + j.replace("/","_") + ".json"] == cluster:
            key_set.append(j)
    print("cluster len: " ,len(key_set))
    random.shuffle(key_set)
    idxs = [i for i in range(len(key_set))]
    if config['si'] != -1:
        si = int(config['si'])
        idxs = []
        for i in range(len(key_set)):
            if i<int(len(key_set)*(0.2*si)) or i>=int(len(key_set)*(0.2*(si+1))):
                idxs.append(i)
    print("idx len:",len(idxs))
    for _ in idxs:
        tmp_time = 0

        par2list=train_set[key_set[_]]
        flag=0
        for l in range(len(output_idx)):
            if float(par2list[output_idx[l]]) <= final_config[l]:
                tmp_time+=par2list[output_idx[l]]
                for k in range(l):
                    tmp_time+=final_config[k]
                time += tmp_time
                flag=1
                break
        if flag == 0:
            time += 2400
            fail += 1
    print("Result of algorithm run: SUCCESS, 0, 0, %f, 0" % time)
    return time


cs = ConfigurationSpace(seed=int(seed))
t_1 = Float("t1", (0, 1200), default=1200)
t_2 = Float("t2", (0, 1200), default=0)
t_3 = Float("t3", (0, 1200), default=0)
t_4 = Float("t4", (0, 1200), default=0)
cluster_ = Categorical("cluster",[cluster],default=cluster)
dataset_ = Categorical("dataset",[dataset],default=dataset)
seed_ = Categorical("seed",[seed],default=seed)
s1_ = Categorical("s1",[s1],default=s1)
s2_ = Categorical("s2",[s2],default=s2)
s3_ = Categorical("s3",[s3],default=s3)
s4_ = Categorical("s4",[s4],default=s4)
si_ = Categorical("si",[start_idx],default=start_idx)
cs.add([t_1, t_2, t_3, t_4,cluster_,dataset_,seed_,s1_,s2_,s3_,s4_,si_])

scenario = Scenario(cs, deterministic=True, n_trials=200)

smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True,w1 = w1,w2= w2)
incumbent = smac.optimize()


output_idx=[]
if cs['s1'] != -1:
    output_idx.append(cs['s1'])
if cs['s2'] != -1:
    output_idx.append(cs['s2'])
if cs['s3'] != -1:
    output_idx.append(cs['s3'])
if cs['s4'] != -1:
    output_idx.append(cs['s4'])
tmp = [incumbent['t1'],incumbent['t2'],incumbent['t3'],incumbent['t4']]
all_ = 0
for i in range(len(output_idx)):
    all_ += tmp[i]
print(str(incumbent['t1']/all_*1200)+","+str(incumbent['t2']/all_*1200)+","+str(incumbent['t3']/all_*1200)+","+str(incumbent['t4']/all_*1200))
