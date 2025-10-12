from os import system
import time as __time
import os
import sys
from functools import partial
from multiprocessing import Pool
from os import popen
import shutil
import random
import logging

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import json
import numpy as np

def normalize(tf,seed):
    with open(tf,'r', encoding='UTF-8') as f:
        fea_dict_ = json.load(f)
    pro_dict = []
    fea_dict = []
    for _ in fea_dict_.keys():
        pro_dict.append(_)
        fea_dict.append(fea_dict_[_])
    fea_dict = np.array(fea_dict)

    max_ = fea_dict.max(axis=0)
    min_ = fea_dict.min(axis=0)


    sub = max_ - min_
    for i in range(len(sub)):
        if sub[i] == 0:
            sub[i] = 1

    new_fea_dict = (fea_dict - min_) / sub
    new_pro_dict = pro_dict

    lim = {"min":list(min_),"sub":list(sub)}
    dict_output = {}
    for _ in range(len(pro_dict)):
        dict_output[pro_dict[_]] = new_fea_dict[_].tolist()
    tf = tf.replace("../","")
    tf = tf.replace("./","")
    tf = tf.replace("/","_")
    with open("tmp/" + tf.replace(".json","_norm" + str(seed) + ".json"),'w', encoding='UTF-8') as f:
        json.dump(dict_output, f)

    with open("tmp/" + tf.replace(".json","_lim" + str(seed) + ".json"),'w', encoding='UTF-8') as f:
        json.dump(lim, f)

def cluster(tfnorm, seed = 0, cluster_num = 20):
    amount_initial_centers = 3
    if amount_initial_centers > cluster_num:
        amount_initial_centers = cluster_num

    with open(tfnorm,'r', encoding='UTF-8') as f:
        fea_dict = json.load(f)

    feature_mat = []
    key_set = []
    for key in fea_dict.keys():
        key_set.append(key)
        feature_mat.append(fea_dict[key])

    print("Read over")
    feature_mat = np.array(feature_mat)
    print(feature_mat.shape)

    train_dict = {}

    X_train = feature_mat

    initial_centers = kmeans_plusplus_initializer(X_train, amount_initial_centers, random_state=seed).initialize()
    xmeans_instance = xmeans(X_train, initial_centers=initial_centers, kmax=cluster_num, ccore=False, random_state=seed)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()

    cluster_center = {"center":list(centers)}

    for j in range(len(centers)):
        for i in range(len(clusters[j])):
            train_dict[key_set[clusters[j][i]]] = j
    with open(tfnorm.replace("_norm"+ str(seed) + ".json","_train_" + str(seed) + ".json"),'w',encoding='utf-8') as f:
        json.dump(train_dict, f)
    with open(tfnorm.replace("_norm"+ str(seed) + ".json","_cluster_center_" + str(seed) + ".json"),'w',encoding='utf-8') as f:
        json.dump(cluster_center, f)

def getTestPortfoliio(tfnorm, clusterPortfolio, solverlist, dataset, seed, outputfile = ""):
    with open(tfnorm,'r', encoding='UTF-8') as f:
        fea_dict = json.load(f)

    with open(clusterPortfolio,'r', encoding='UTF-8') as f:
        output_dict = json.load(f)
    portfolio_dict = output_dict['portfolio']
    center_dict = output_dict['center']
    portfolio_ = {}
    time_ = {}
    for key in portfolio_dict.keys():
        tmp = portfolio_dict[key]
        time_[key] = tmp[1]
        solverlist_ = tmp[0]
        outputsolver = []
        for i in range(len(solverlist_)):
            outputsolver.append(solverlist[solverlist_[i]])
        portfolio_[key] = outputsolver

    feature_mat = []
    key_set = []
    for key in fea_dict.keys():
        key_set.append(key)
        feature_mat.append(fea_dict[key])
    centers = center_dict['center']
    X_test = np.array(feature_mat)
    test_dict = {}
    for j in range(len(X_test)):
        dist = []
        for i in range(len(centers)):
            dist.append(np.sqrt(np.sum((X_test[j] - np.array(centers[i]))**2)))

        index = np.argmin(dist)
        tmp = [portfolio_[str(index)],time_[str(index)]]
        test_dict[key_set[j]] = tmp

    dirpath = 'output'
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    if outputfile == "":
        outputfile = "output/test_result_" + str(dataset) + "_" + str(seed) + "_" + str(len(centers)) + ".json"

    with open(outputfile,'w',encoding='utf-8') as f:
        json.dump(test_dict, f)

def RunSeed3(sf, seed,start_idx):
    command = "python -u portfolio_smac3.py " + " -seed " + str(seed)
    for key in sf[0].keys():
        command = command + " -" + str(key) + " " + str(sf[0][key])
    command = command + " -si "+str(start_idx)
    print(command)
    output = popen(command).read()
    tmp = output.split('\n')
    return [tmp,sf[1]]

def get_portfolio_3(solverlist, td, tc, tlim, tcenter, dataset, outputfile = "", portfolioSize = 4, cluster_num = 20, seed = 0, timelimit = 1200):
    with open(td,'r', encoding='UTF-8') as f:
        par2_dict = json.load(f)
    with open(tc,'r', encoding='UTF-8') as f:
        train_cluster_dict = json.load(f)

    with open(tlim,'r', encoding='UTF-8') as f:
        lim_dict = json.load(f)
    with open(tcenter,'r', encoding='UTF-8') as f:
        center_dict = json.load(f)

    if portfolioSize > len(solverlist):
        print("warning: PortfolioSize is bigger than the number of solvers!")
        PortfolioSize = len(solverlist)


    dirpath = 'output'
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    if outputfile == "":
        outputfile = "output/train_result_" + str(dataset) + "_" + str(portfolioSize) + "_" + str(cluster_num) + "_" + str(seed) + ".json"

    cluster = 0
    final_portfolio = {}

    for cluster in range(cluster_num):
        print(solverlist)
        cov = [0 for i in range(len(solverlist))]


        output_idx = []

        final_config = []

        train_set = par2_dict['train']

        for i in range(portfolioSize):
            min_time = 0
            min_idx = -1
            sf = []
            pf = []
            for j in range(len(solverlist)):
                tmpdict = {}
                if cov[j] == 1:
                    continue
                tmp = []
                for l in output_idx:
                    tmp.append(l)
                tmp.append(j)
                tmpdict['t1'] = 1200
                tmpdict['t2'] = 0
                tmpdict['t3'] = 0
                for l in range(len(tmp)):
                    tmpdict["s"+str(l+1)] = str(tmp[l])
                tmpdict["cluster"] = cluster
                if dataset == 'Equality+LinearArith':
                    tmpdict["dataset"] = "ELA"
                else:
                    tmpdict["dataset"] = str(dataset)

                sf.append([tmpdict,j])
            valid_scores = [0 for i in range(len(sf))]
            for si in range(0,5):
                p = Pool(processes=10)

                partial_RunSeed = partial(RunSeed3, seed = seed,start_idx=si)
                ret = p.map(partial_RunSeed, sf)

                p.close()
                p.join()

                ret_seed = []
                for l in range(len(ret)):
                    ret_seed.append(ret[l][1])
                    k = ret[l][0][-2].split(",")
                    ret[l] = k
                print(ret)
                configs = [[] for i in range(len(ret))]
                for l in range(len(ret)):
                    tmp_config = []
                    for _ in range(len(ret[l])):
                        tmp_config.append(float(ret[l][_]))
                    configs[l] = tmp_config

                key_set = list(train_set.keys())

                full_key_set = list(train_set.keys())
                key_set = []

                if dataset == "Equality+LinearArith":
                    dataplace = "ELA"
                elif dataset == "QF_Bitvec":
                    dataplace = "QFBV"
                elif dataset == "QF_NonLinearIntArith":
                    dataplace = "QFNIA"
                else:
                    dataplace = dataset
                for j in full_key_set:
                    if j in train_cluster_dict.keys() and train_cluster_dict[j] == cluster:
                        key_set.append(j)
                    if  "./infer_result/"+str(dataset)+"/_data_sibly_sibyl_data_"+str(dataset)+"_"+str(dataset)+"_" + j.replace("/","_") + ".json" in train_cluster_dict.keys() and train_cluster_dict["./infer_result/"+str(dataset)+"/_data_sibly_sibyl_data_"+str(dataset)+"_"+str(dataset)+"_" + j.replace("/","_") + ".json"] == cluster:
                        key_set.append(j)
                    if  "./infer_result/"+str(dataplace)+"/_data_sibly_sibyl_data_Comp_non-incremental_" + j.replace("/","_") + ".json" in train_cluster_dict.keys() and train_cluster_dict["./infer_result/"+str(dataplace)+"/_data_sibly_sibyl_data_Comp_non-incremental_" + j.replace("/","_") + ".json"] == cluster:
                        key_set.append(j)

                print("configs len:",len(configs))
                print(output_idx)

                for config_idx in range(len(configs)):
                    config = configs[config_idx]
                    x1 = config[0]
                    x2 = config[1]
                    x3 = config[2]
                    x4 = 1200-x1-x2-x3
                    tmp_config = [x1,x2,x3,x4]

                    time = 0

                    ri = int(len(key_set)*(0.2*(si+1)))
                    ri = min(ri,int(len(key_set)))
                    for _ in range(int(len(key_set)*(0.2*si)),ri):
                        tmp_time = 0
                        flag=0
                        par2list = train_set[key_set[_]]
                        tmplist_ = []
                        for idx in output_idx:
                            tmplist_.append(idx)
                        tmplist_.append(ret_seed[config_idx])
                        for l in range(len(tmplist_)):
                            if float(par2list[tmplist_[l]]) <= tmp_config[l]:
                                tmp_time+=par2list[tmplist_[l]]
                                for k in range(l):
                                    tmp_time+=tmp_config[k]
                                time += tmp_time
                                flag=1
                                break
                        if flag == 0:
                            time += 2400
                    valid_scores[config_idx] += time
            print(valid_scores)
            chosen_idx = np.argmin(valid_scores)
            final_config = configs[chosen_idx]
            output_idx.append(ret_seed[chosen_idx])
            cov[ret_seed[chosen_idx]] = 1

        tmpdict = {}
        tmpdict['t1'] = 1200
        tmpdict['t2'] = 0
        tmpdict['t3'] = 0
        for l in range(len(output_idx)):
            tmpdict["s"+str(l+1)] = str(output_idx[l])
        tmpdict["cluster"] = cluster
        if dataset == 'Equality+LinearArith':
            tmpdict["dataset"] = "ELA"
        else:
            tmpdict["dataset"] = str(dataset)
        sf = []
        sf.append([tmpdict,-1])
        p = Pool(processes=1)
        partial_RunSeed = partial(RunSeed3, seed = seed,start_idx=-1)
        ret = p.map(partial_RunSeed, sf)
        p.close()
        p.join()
        for l in range(len(ret)):
            k = ret[l][0][-2].split(",")
            ret[l] = k
        print(ret)
        configs = [[] for i in range(len(ret))]
        for l in range(len(ret)):
            tmp_config = []
            for _ in range(len(ret[l])):
                tmp_config.append(float(ret[l][_]))
            configs[l] = tmp_config
        final_config = configs[0]
        final_portfolio[cluster] = [output_idx,final_config]
    output_dict = {"portfolio":final_portfolio,"lim":lim_dict,"center":center_dict}
    with open(outputfile,'w',encoding='utf-8') as f:
        json.dump(output_dict, f)

if __name__ == '__main__':
    work_type = 'infer'
    if sys.argv[1] == 'train' or sys.argv[1] == 'infer':
        work_type = sys.argv[1]
    tf = ""
    td = ""
    seed = 0
    cluster_num = 20
    solverdict = ""
    dataset = ""
    clusterPortfolio = ""
    for i in range(len(sys.argv)-1):
        if (sys.argv[i] == '-train_features'):
            tf = sys.argv[i+1]
        if (sys.argv[i] == '-train_data'):
            td = sys.argv[i+1]

        if (sys.argv[i] == '-seed'):
            seed = int(sys.argv[i+1])
        if (sys.argv[i] == '-cluster_num'):
            cluster_num = int(sys.argv[i+1])

        if (sys.argv[i] == '-solverdict'):
            solverdict = sys.argv[i+1]

        if (sys.argv[i] == '-dataset'):
            dataset = sys.argv[i+1]


        if (sys.argv[i] == '-clusterPortfolio'):
            clusterPortfolio = sys.argv[i+1]

    tf = "./machfea/infer_result/" + str(dataset) + "_train_feature.json"
    td = "./data/" + str(dataset) + "Labels.json"
    dirpath = 'tmp'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    if work_type == "train":
        ### normalize ###
        normalize(tf,seed)
        ### clustering ###
        tmp = tf
        tmp = tmp.replace("../","")
        tmp = tmp.replace("./","")
        tmp = tmp.replace("/","_")
        tfnorm = "tmp/"+tmp.replace(".json","_norm" + str(seed) + ".json")
        tflim = tfnorm.replace("_norm","_lim")
        tcenter = tfnorm.replace("_norm" + str(seed) + ".json","_cluster_center_" + str(seed) + ".json")
        cluster(tfnorm,seed,cluster_num)
        tc = tfnorm.replace("_norm" + str(seed) + ".json","_train_" + str(seed) + ".json")
        ### scheduling ###
        with open(solverdict,'r', encoding='UTF-8') as f:
            solver_dict = json.load(f)
        solverlist = solver_dict["solver_list"]
        get_portfolio_3(solverlist, td, tc, tflim, tcenter, dataset, outputfile = "", portfolioSize = 4, cluster_num = cluster_num, seed = seed, timelimit = 1200)

    elif work_type == "infer":
        with open(clusterPortfolio,'r', encoding='UTF-8') as f:
            output_dict = json.load(f)
        lim = output_dict['lim']
        min_ = lim['min']
        sub = lim['sub']
        testf = "./machfea/infer_result/" + str(dataset) + "_test_feature.json"

        with open(testf,'r', encoding='UTF-8') as f:
            fea_dict_ = json.load(f)
        pro_dict = []
        fea_dict = []
        for _ in fea_dict_.keys():
            pro_dict.append(_)
            fea_dict.append(fea_dict_[_])
        fea_dict = np.array(fea_dict)
        print(fea_dict.shape)

        new_fea_dict = (fea_dict - min_) / sub
        new_pro_dict = pro_dict

        print(len(new_fea_dict))
        print(len(new_pro_dict))
        dict_output = {}
        for _ in range(len(pro_dict)):
            dict_output[pro_dict[_]] = new_fea_dict[_].tolist()
        testf = testf.replace("../","")
        testf = testf.replace("./","")
        testf = testf.replace("/","_")
        testnorm = "tmp/" + testf.replace(".json","_norm" + str(seed) + ".json")
        with open(testnorm,'w', encoding='UTF-8') as f:
            json.dump(dict_output, f)
        with open(solverdict,'r', encoding='UTF-8') as f:
            solver_dict = json.load(f)
        solverlist = solver_dict["solver_list"]
        getTestPortfoliio(testnorm,clusterPortfolio,solverlist,dataset,seed)
