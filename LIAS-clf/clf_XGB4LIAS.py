# this code is for the model alternatives with classification task
    # random forest, xgboost, deep learning model: MLP
# each task train a model. the output of model can not be weighted


# classification task:
# input: data, workload
# output: the best-performing index algorithm

import os,csv
import numpy as np
import pandas as pd
import json,joblib
import hyperopt
import xgboost as xgb
import warnings
import time
import random
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

from util import generate_clf_header,run_clf_xgb_hypertuning,default_dump,write_res,select_naive_workload
from util import run_clf_RF_hypertuning,generate_val_clf_header


def generate_three_weights():
    # generate two random value，sort
    num1 = random.random()
    num2 = random.random()
    num1, num2 = sorted([num1, num2])

    # produce three weights, and shuffle
    a = num1
    b = num2 - num1
    c = 1 - num2
    weights = [a,b,c]
    random.shuffle(weights)

    return weights


# generate constraint for objective
def generate_constraint(labels,c_num,c_seed = None):
    # 
    train_data = '/data/cloudGroup/xxxx/index_selection/new_test_adaptation/total_feature_all_29851'
    reg_data = pd.read_csv(train_data)
    total_constraints = []
    total_metric_min = []
    total_metric_max = []
    metric_min = {}
    metric_max = {}
    if c_seed != None:
        random.seed(c_seed)
    for label in labels:
        label_min = reg_data[label].min()
        label_max = reg_data[label].max()
        metric_min[label] = label_min
        metric_max[label] = label_max
    for i in range(c_num):
        constraints = {}
        for label in labels:
            label_min = reg_data[label].min()
            label_max = reg_data[label].max()
            constraints[label] = random.uniform(label_min,label_max)
        total_constraints.append(constraints)
        total_metric_min.append(metric_min)
        total_metric_max.append(metric_max)
    return total_constraints,total_metric_min, total_metric_max


#generate objective's constraint according to performance value
def generate_groupby_constraint(labels,intersect_df,c_seed = None):
    # group by according to dataname and opsname
    # train_data = '/home/xxx/index_selection/new_test_adaptation/total_feature_all_29851'
    train_data = '/data/xxx/index_selection/new_test_adaptation/total_feature_all_29851'
    reg_data = pd.read_csv(train_data)
    groupdata = reg_data.groupby(['dataname','opsname'])

    total_constraints = []

    metric_min = {}
    metric_max = {}
    if c_seed != None:
        random.seed(c_seed)
    for label in labels:
        label_min = reg_data[label].min()
        label_max = reg_data[label].max()
        metric_min[label] = label_min
        metric_max[label] = label_max
    for i, row in intersect_df.iterrows():
        dataname = row['dataname']
        opsname = row['opsname']
        subdata = groupdata.get_group((dataname,opsname))
        constraints = {}
        for label in labels:
            label_min = subdata[label].min()
            label_max = subdata[label].max()
            constraints[label] = random.uniform(label_min,label_max)
        total_constraints.append(constraints)

    return total_constraints,metric_min, metric_max



def split_train_test(train_ddf,clfheader,label,self_seed = None,valid_size=0.1,
                     test_size=0.1, split_type="random"):
    train_ddf = train_ddf[clfheader]
    label_values = train_ddf['indexname'].unique()
    num_class = len(label_values)
    li = 0
    class_name = {}
    for x in label_values:
        train_ddf.loc[train_ddf['indexname'] == x, 'indexname'] = li
        class_name[x] = li
        li += 1
    # diagnoses = {'negative': 0,
    #              'hypothyroid': 1,
    #              'hyperthyroid': 2}
    #
    # xgbDF['target'] = xgbDF['target'].map(diagnoses)

    train_ddf[['indexname']] = train_ddf[['indexname']].astype('int')
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if self_seed != None:
        np.random.seed(self_seed)
    else:
        np.random.seed()
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()

    y_train = train_ddf['indexname']
    y_test = test_ddf['indexname']
    y_valid = valid_ddf['indexname']


    x_train = train_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_test = test_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_valid = valid_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_train = x_train.drop(['indexname',label], axis=1)
    x_test = x_test.drop(['indexname',label], axis=1)
    x_valid = x_valid.drop(['indexname',label], axis=1)

    # num_class = len((y_train['indexname'].unique()))
    return  x_train,y_train,x_test,y_test,x_valid,y_valid,num_class,class_name

def clf_data_normalize(train_ddf):
    divide_big_num_name = [ 'datasize']  # divide 1 million 1000000
    log_column_name = [  'max_key','minkey',
                         'pwl_num4', 'max_size4', 'min_size4', 'max_gap4', 'min_gap4',
                         'pwl_num8', 'max_size8', 'min_size8', 'max_gap8', 'min_gap8',
                         'pwl_num16', 'max_size16', 'min_size16', 'max_gap16', 'min_gap16',
                         'pwl_num32', 'max_size32', 'min_size32', 'max_gap32', 'min_gap32',
                         'pwl_num64', 'max_size64', 'min_size64', 'max_gap64', 'min_gap64',
                         'pwl_num128', 'max_size128', 'min_size128', 'max_gap128', 'min_gap128',
                         'pwl_num256', 'max_size256', 'min_size256', 'max_gap256', 'min_gap256',
                         'pwl_num512', 'max_size512', 'min_size512', 'max_gap512', 'min_gap512',
                         'pwl_num1024', 'max_size1024', 'min_size1024', 'max_gap1024', 'min_gap1024',
                         'pwl_num2048', 'max_size2048', 'min_size2048', 'max_gap2048', 'min_gap2048',
                         'pwl_num4096', 'max_size4096', 'min_size4096', 'max_gap4096', 'min_gap4096']
    # multiply_hundred_name = ['rq','nl','i','hotratio']
    multiply_hundred_name = ['rq','i']


    for c in divide_big_num_name:
        # value = train_ddf[c]
        train_ddf[c] = train_ddf[c].div(1000000)

    for c in log_column_name:
        # print(X[c])
        train_ddf[c] = train_ddf[c].apply(np.log1p)

    for c in multiply_hundred_name:

        train_ddf.loc[:,c] = train_ddf.loc[:,c]*100


    # print('Jesus, please be with Brother gao')

    return train_ddf




def split_retrain_test(train_ddf,self_seed = None,valid_size=0.1,
                     test_size=0.1, split_type="random"):

    label_values = train_ddf['indexname'].unique()
    num_class = len(label_values)
    li = 0
    class_name = {}
    # for x in label_values:
    #     train_ddf.loc[train_ddf['indexname'] == x, 'indexname'] = li
    #     class_name[x] = li
    #     li += 1
    # diagnoses = {'negative': 0,
    #              'hypothyroid': 1,
    #              'hyperthyroid': 2}
    #
    # xgbDF['target'] = xgbDF['target'].map(diagnoses)

    train_ddf[['indexname']] = train_ddf[['indexname']].astype('int')
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if self_seed != None:
        np.random.seed(self_seed)
    else:
        np.random.seed()
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()

    y_train = train_ddf['indexname']
    y_test = test_ddf['indexname']
    y_valid = valid_ddf['indexname']


    x_train = train_ddf.drop(['index'], axis=1)
    x_test = test_ddf.drop(['index'], axis=1)
    x_valid = valid_ddf.drop(['index'], axis=1)
    x_train = x_train.drop(train_ddf.columns[[0]], axis=1)
    x_test = x_test.drop(train_ddf.columns[[0]], axis=1)
    x_valid = x_valid.drop(train_ddf.columns[[0]], axis=1)
    x_train = x_train.drop(['indexname'], axis=1)
    x_test = x_test.drop(['indexname'], axis=1)
    x_valid = x_valid.drop(['indexname'], axis=1)

    # num_class = len((y_train['indexname'].unique()))
    return  x_train,y_train,x_test,y_test,x_valid,y_valid,num_class,class_name


def init_split_train_test(train_ddf,clfheader,label,self_seed = None,valid_size=0.1,
                     test_size=0.1, split_type="random"):
    train_ddf = train_ddf[clfheader]
    label_values = train_ddf['indexname'].unique()
    num_class = len(label_values)
    li = 0
    class_name = {}
    for x in label_values:
        train_ddf.loc[train_ddf['indexname'] == x, 'indexname'] = li
        class_name[x] = li
        li += 1
    # diagnoses = {'negative': 0,
    #              'hypothyroid': 1,
    #              'hyperthyroid': 2}
    #
    # xgbDF['target'] = xgbDF['target'].map(diagnoses)

    train_ddf[['indexname']] = train_ddf[['indexname']].astype('int')
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if self_seed != None:
        np.random.seed(self_seed)
    else:
        np.random.seed()
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()

    y_train = train_ddf['indexname']
    y_test = test_ddf['indexname']
    y_valid = valid_ddf['indexname']


    x_train = train_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_test = test_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_valid = valid_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_train = x_train.drop([label], axis=1)
    x_test = x_test.drop([label], axis=1)
    x_valid = x_valid.drop([label], axis=1)

    # num_class = len((y_train['indexname'].unique()))
    return  x_train,y_train,x_test,y_test,x_valid,y_valid,num_class,class_name


def adapt_split_train_test(train_ddf,clfheader,label,init_class_name,self_seed = None,valid_size=0.1,
                     test_size=0.1, split_type="random"):
    train_ddf = train_ddf[clfheader]
    label_values = train_ddf['indexname'].unique()
    num_class = len(label_values)

    train_ddf['indexname'] = train_ddf['indexname'].map(init_class_name)

    train_ddf[['indexname']] = train_ddf[['indexname']].astype('int')
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if self_seed != None:
        np.random.seed(self_seed)
    else:
        np.random.seed()
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()

    y_train = train_ddf['indexname']
    y_test = test_ddf['indexname']
    y_valid = valid_ddf['indexname']


    x_train = train_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_test = test_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_valid = valid_ddf.drop(train_ddf.columns[[0]], axis=1)
    x_train = x_train.drop(['indexname',label], axis=1)
    x_test = x_test.drop(['indexname',label], axis=1)
    x_valid = x_valid.drop(['indexname',label], axis=1)

    # num_class = len((y_train['indexname'].unique()))
    return  x_train,y_train,x_test,y_test,x_valid,y_valid,label_values,num_class



# 117 path
clf_datapath = '/data/xxx/index_selection/classification_data/'
weight_clf_datapath = '/data/xxx/index_selection/weighted_clf_data/'
val_datapath = '/data/xxx/index_selection/validation_data/'
original_val_datapath = '/data/xxx/index_selection/original_validation_data/'
reg_datapath = '/data/xxx/index_selection/data_ops_index_label/total_feature'

respath = './result/Alternatives_Clf/'


def load_model(modelpath,label):

    assert (os.path.exists(modelpath)), label+" model does not exist"
    reg_mod = joblib.load(modelpath)

    return reg_mod


def compute_single_regret(label,total_valdata, val_preds, valdata_y, allsampling = 'systematic_sampling_10thousand'):

    total_regret = 0
    not_find = 0
    for i, row in total_valdata.iterrows():
        dataname = row['dataname']
        opsname = row['opsname']
        val_name = dataname+'_'+opsname.split('_')[0]+allsampling
        valpath = os.path.join(val_datapath,label)
        valfile = os.path.join(valpath,val_name)
        data = pd.read_csv(valfile)
        if val_preds[i] == valdata_y[i]:
            regret = 0
        else:
            indexdata = data['indexname'].tolist()
            if (val_preds[i] in indexdata ):
                # rank = indexdata.index(val_preds[i])
                # regret = rank / 13
                select_x = val_preds[i]
                rank = indexdata.index(select_x)
                x_value = data[label].iloc[rank]
                regret = (1-(min(data[label].iloc[0],x_value)/max(data[label].iloc[0],x_value)))
            else:
                predict_x = val_preds[i]
                regret = judge_available (dataname,predict_x,label) # 判断index 是否可用
                if regret == 0:
                    # print ('Lord, I needs You!')
                    not_find += 1
                # print (dataname,',',opsname,',',val_preds[i])
                # print(val_preds[i])
            # pred 在第几位
        total_regret += regret

    

    return total_regret,not_find

def constraint_compute_regret(objective,con_label,total_constraint,total_valdata, val_preds, valdata_y, allsampling = 'systematic_sampling_10thousand'):

    total_regret = 0
    not_find = 0
    for i, row in total_valdata.iterrows():
        dataname = row['dataname']
        opsname = row['opsname']
        val_name = dataname+'_'+opsname.split('_')[0]+allsampling
        valpath = os.path.join(original_val_datapath,objective)
        valfile = os.path.join(valpath,val_name)
        data = pd.read_csv(valfile)
        constraint = total_constraint[i][con_label]
        select_x = val_preds[i]
        # 判断是否有索引能够满足constraint

        if con_label == 'throughput':
            max_con = data[con_label].max()
            if max_con < constraint:
                print('Lord, no index satisify constraint')
                not_find += 1
                regret = 0
                continue
        else:
            min_con = data[con_label].min()
            if min_con > constraint:
                print('Lord, no index satisify constraint')
                not_find += 1
                regret = 0
                continue
        indexdata = data['indexname'].tolist()
        if select_x in indexdata:
            regret = judge_available(dataname,select_x,objective) # judge index whether applicable
            if regret == 0:
                # judge the select_x whether satisgy constraint
                select_performance = data.loc[data['indexname'] == select_x]
                if con_label == 'throughput':
                    a = select_performance[con_label].values[0]
                    if a < constraint:
                        regret = 1
                    else:# 计算 regret
                        con_data = data[data[con_label] >= constraint]
                        if objective == 'throughput':
                            sortvalue = con_data.sort_values(by=objective,inplace=False,ascending=False)
                        else:
                            sortvalue = con_data.sort_values(by=objective,inplace=False,ascending=True)
                        indexdata = sortvalue['indexname'].tolist()
                        best_x = indexdata[0]
                        if select_x == best_x:
                            regret = 0
                        else:
                            rank = indexdata.index(select_x)
                            x_value = sortvalue[objective].iloc[rank]
                            best_metric = sortvalue[objective].iloc[0]
                            # if (x_value * best_metric<0) :
                            v0= abs(x_value/best_metric)
                            v1 = abs(best_metric/x_value)
                            regret = (1-min(v0,v1))
                else:
                    a = select_performance[con_label].values[0]
                    if a > constraint:
                        regret = 1
                    else:# 计算 regret
                        con_data = data[data[con_label] <= constraint]
                        if objective == 'throughput':
                            sortvalue = con_data.sort_values(by=objective,inplace=False,ascending=False)
                        else:
                            sortvalue = con_data.sort_values(by=objective,inplace=False,ascending=True)
                        indexdata = sortvalue['indexname'].tolist()
                        best_x = indexdata[0]
                        if select_x == best_x:
                            regret = 0
                        else:
                            rank = indexdata.index(select_x)
                            x_value = sortvalue[objective].iloc[rank]
                            best_metric = sortvalue[objective].iloc[0]
                            # if (x_value * best_metric<0) :
                            v0= abs(x_value/best_metric)
                            v1 = abs(best_metric/x_value)
                            regret = (1-min(v0,v1))
        else:
            regret = 0
            not_find += 1
        total_regret += regret



    return total_regret,not_find


def multi_constraint_compute_regret(objective,con_labels,total_constraint,total_valdata, val_preds, valdata_y, allsampling = 'systematic_sampling_10thousand'):

    total_regret = 0
    not_find = 0
    for i, row in total_valdata.iterrows():
        dataname = row['dataname']
        opsname = row['opsname']
        val_name = dataname+'_'+opsname.split('_')[0]+allsampling
        valpath = os.path.join(original_val_datapath,objective)
        valfile = os.path.join(valpath,val_name)
        data = pd.read_csv(valfile)
        for con_label in con_labels:
            constraint = total_constraint[i][con_label]
            select_x = val_preds[i]
            # judge any index can satisgy constraint

            if con_label == 'throughput':
                max_con = data[con_label].max()
                if max_con < constraint:
                    print('Lord, no index satisify constraint')
                    not_find += 1
                    regret = 0
                    continue
            else:
                min_con = data[con_label].min()
                if min_con > constraint:
                    print('Lord, no index satisify constraint')
                    not_find += 1
                    regret = 0
                    continue
            indexdata = data['indexname'].tolist()
            if select_x in indexdata:
                regret = judge_available(dataname,select_x,objective) 
                if regret == 0:
                    # whether select_x satisfy constraint
                    select_performance = data.loc[data['indexname'] == select_x]
                    if con_label == 'throughput':
                        a = select_performance[con_label].values[0]
                        if a < constraint:
                            regret = 1
                        else:# compute regret
                            con_data = data[data[con_label] >= constraint]
                            if objective == 'throughput':
                                sortvalue = con_data.sort_values(by=objective,inplace=False,ascending=False)
                            else:
                                sortvalue = con_data.sort_values(by=objective,inplace=False,ascending=True)
                            indexdata = sortvalue['indexname'].tolist()
                            best_x = indexdata[0]
                            if select_x == best_x:
                                regret = 0
                            else:
                                rank = indexdata.index(select_x)
                                x_value = sortvalue[objective].iloc[rank]
                                best_metric = sortvalue[objective].iloc[0]
                                # if (x_value * best_metric<0) :
                                v0= abs(x_value/best_metric)
                                v1 = abs(best_metric/x_value)
                                regret = (1-min(v0,v1))
                    else:
                        a = select_performance[con_label].values[0]
                        if a > constraint:
                            regret = 1
                        else:# compute regret
                            con_data = data[data[con_label] <= constraint]
                            if objective == 'throughput':
                                sortvalue = con_data.sort_values(by=objective,inplace=False,ascending=False)
                            else:
                                sortvalue = con_data.sort_values(by=objective,inplace=False,ascending=True)
                            indexdata = sortvalue['indexname'].tolist()
                            best_x = indexdata[0]
                            if select_x == best_x:
                                regret = 0
                            else:
                                rank = indexdata.index(select_x)
                                x_value = sortvalue[objective].iloc[rank]
                                best_metric = sortvalue[objective].iloc[0]
                                # if (x_value * best_metric<0) :
                                v0= abs(x_value/best_metric)
                                v1 = abs(best_metric/x_value)
                                regret = (1-min(v0,v1))
            else:
                regret = 0
                not_find += 1
            total_regret += regret



    return total_regret,not_find


def judge_available(data_name, select_x,label = 'indexsize'):
    workload_feature = select_naive_workload(data_name)
    if ('wiki' in data_name or 'lognormal' in data_name):
        is_duplicate = True
    else:
        is_duplicate = False
    if ('_1m_' in data_name ):
        is_delta = True
    else:
        is_delta = False
    if is_duplicate:# if duplicate data
        not_applicable = ['FAST','LIPP','FINEdex','ART','Wormhole','XIndex','HOT']
    else:
        not_applicable = []

    # workload = [ops_num,rq_ratio,nl_ratio,insert_ratio,insert_type,hotratio,mix,zipfratio,thread_num]
    # if insert? workload []
    if workload_feature[3] > 0:
        # print('the workload contain insert ~ ~ i need You, my Lord' )
        non_insert = ['RMI','PGM','FAST','TS']
        not_applicable += non_insert
    else:
        insert = ['DynamicPGM']
        not_applicable += insert
    # if multi-thread
    if workload_feature[8] > 1:
        # print('the workload is multi-thread ~ ~ please come, my Lord')
        single_thread = ['RMI','ALEX','BTree','MABTree','FAST','PGM','LIPP','DynamicPGM','ART','DILI','TS','HOT']
        not_applicable += single_thread
    else:
        multi_thread = ['ARTOLC']
        not_applicable += multi_thread
    if is_delta:
        delta = ['FINEdex']
        not_applicable += delta
    if label == 'bulkloadtime':
        bulk = ['DILI']
        not_applicable += bulk
    not_applicable = list(set(not_applicable))
    if select_x in not_applicable:
        return 1
    else:
        return 0

def accuracy(true_y, predict_y):
    cnt1 = 0
    cnt2 = 0
    for i in range(len(predict_y)):
        if predict_y[i] == true_y[i]:
            cnt1 += 1
        else:
            cnt2 += 1
    # print ('Lord, I needs You, we train the clf4LIAS')
    all_accuracy = cnt1 / (cnt1 + cnt2)
    return all_accuracy


def res_dict_write(label,val_data,modeltype,feature_type,sampling,feature_num,trainingtime,clf_model,
                   adapt_v,init_x_num,incremental_x_num,valdata_accuracy,vald_reference_time,
                   clf_regret,intersect_num,avg_regret,not_found,max_c_80,acc_score):
    res_dict = {}
    res_dict['task'] = label
    res_dict['data'] = val_data
    res_dict['model'] = modeltype
    res_dict['feature'] = feature_type
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = trainingtime
    if  modeltype.find('clfXGB') != -1:
        res_dict['n_estimators'] = clf_model.n_estimators
        res_dict['learning_rate'] = clf_model.learning_rate
        res_dict['max_depth'] = clf_model.max_depth
        res_dict['min_child_weight'] = clf_model.min_child_weight
    elif  modeltype.find('clfRF') != -1 :
        res_dict['n_estimators'] = clf_model.n_estimators
        res_dict['min_samples_split'] = clf_model.min_samples_split
        res_dict['max_depth'] = clf_model.max_depth
        res_dict['min_samples_leaf'] = clf_model.min_samples_leaf
    res_dict['adapter'] = adapt_v
    res_dict['init_data_num'] = init_x_num
    res_dict['adapt_data_num'] = incremental_x_num
    res_dict['accuracy'] = valdata_accuracy
    res_dict['reference_time'] = vald_reference_time
    res_dict['regret'] = clf_regret
    res_dict['intersect_num'] = intersect_num
    res_dict['avg_regret'] = avg_regret
    res_dict['not_found'] = not_found
    res_dict['max_c_80'] = max_c_80
    res_dict['acc_score'] = acc_score

    return res_dict


def constraint_res_dict_write(objective,val_data,modeltype,feature_type,sampling,feature_num,trainingtime,
                   adapt_v,init_x_num,incremental_x_num,valdata_accuracy,vald_reference_time,
                   clf_regret,intersect_num,avg_regret,not_found,con_label,constraint,c_seed):
    res_dict = {}
    res_dict['task'] = objective
    res_dict['data'] = val_data
    res_dict['model'] = modeltype
    res_dict['feature'] = feature_type
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = trainingtime
    res_dict['adapter'] = adapt_v
    res_dict['init_data_num'] = init_x_num
    res_dict['adapt_data_num'] = incremental_x_num
    res_dict['accuracy'] = valdata_accuracy
    res_dict['reference_time'] = vald_reference_time
    res_dict['regret'] = clf_regret
    res_dict['intersect_num'] = intersect_num
    res_dict['avg_regret'] = avg_regret
    res_dict['not_found'] = not_found
    res_dict['con_label'] = con_label
    res_dict['constraint'] = constraint
    res_dict['c_seed'] = c_seed

    return res_dict


def add_new_feature_rq_i(train_ddf):
    i_pos = train_ddf.columns.get_loc('i')
    rq_pos = train_ddf.columns.get_loc('rq')
    train_ddf.insert(i_pos,'is_i',0)
    train_ddf.insert(rq_pos,'is_rq',0)
    thread_pos = train_ddf.columns.get_loc('thread')
    train_ddf.insert(thread_pos,'lookup',0)
    train_ddf['lookup'] = train_ddf.apply(lambda x:1-(x['i']+x['rq']),axis = 1)
    train_ddf.loc[(train_ddf.rq>0),'is_rq'] = 1
    train_ddf.loc[(train_ddf.i>0),'is_i'] = 1

    return train_ddf

def XGB_Clf(label,allsampling,feature_type,reg_data_num,v_data_num,sampling,val_data = 'finish',
            modeltype = 'clfXGB',adapt_v=None,test_train = True):
    print('Lord, please be with us, today is Jan 4')
    print('Lord, this is XGB for classification task: ', label)

    # 划分训练集和测试集
    # load clf data
    if adapt_v:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num)+'_'+adapt_v)
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)
    else:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num))
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)

    # 加入特征 is_i, is_rq, lookup
    clfdata = add_new_feature_rq_i(clfdata)

    train_unique_class = [1,0]
    all_unique_class = [0,2]
    while (set(train_unique_class) != (set(all_unique_class))):
        x_train,y_train,x_test,y_test,x_valid,y_valid,num_class,class_name = split_train_test(clfdata,clfheader,label)
        train_unique_class = y_train.unique()
        all_unique_class = list(range(0,num_class))
        x_train = clf_data_normalize(x_train)
        x_test = clf_data_normalize(x_test)
        x_valid = clf_data_normalize(x_valid)
        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)




    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']]
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)
    if feature_type == "naiveDHfeature":
        drop_DH_featurename = ['datasize', 'max_key','minkey',
                               'pwl_num4', 'max_size4', 'min_size4', 'max_gap4', 'min_gap4',
                               'pwl_num8', 'max_size8', 'min_size8', 'max_gap8', 'min_gap8',
                               'pwl_num16', 'max_size16', 'min_size16', 'max_gap16', 'min_gap16',
                               'max_size32', 'min_size32', 'max_gap32', 'min_gap32',
                               'pwl_num64', 'max_size64', 'min_size64', 'max_gap64', 'min_gap64',
                               'pwl_num128', 'max_size128', 'min_size128', 'max_gap128', 'min_gap128',
                               'pwl_num256', 'max_size256', 'min_size256', 'max_gap256', 'min_gap256',
                               'pwl_num512', 'max_size512', 'min_size512', 'max_gap512', 'min_gap512',
                               'pwl_num1024', 'max_size1024', 'min_size1024', 'max_gap1024', 'min_gap1024',
                               'pwl_num2048', 'max_size2048', 'min_size2048', 'max_gap2048', 'min_gap2048',
                               'max_size4096', 'min_size4096', 'max_gap4096', 'min_gap4096']
        x_train = x_train.drop(drop_DH_featurename, axis=1)
        x_test = x_test.drop(drop_DH_featurename, axis=1)
        x_valid = x_valid.drop(drop_DH_featurename, axis=1)
    elif feature_type == "naiveWorkloadFeature":

        drop_w_fearture = []
        for i in range(32):
            drop_w_fearture.append('em'+str(i))
        other_drop = ['is_rq','is_i','rq','i','lookup','thread','ops_num']
        drop_w_fearture += other_drop
        x_train = x_train.drop(drop_w_fearture, axis=1)
        x_test = x_test.drop(drop_w_fearture, axis=1)
        x_valid = x_valid.drop(drop_w_fearture, axis=1)
        print('Lord, I needs You!')
    elif feature_type == "naiveDH_naiveWorkload":
        drop_w_fearture = []
        for i in range(32):
            drop_w_fearture.append('em'+str(i))
        other_drop = ['is_rq','is_i','rq','i','lookup','thread','ops_num']
        drop_w_fearture += other_drop
        x_train = x_train.drop(drop_w_fearture, axis=1)
        x_test = x_test.drop(drop_w_fearture, axis=1)
        x_valid = x_valid.drop(drop_w_fearture, axis=1)
        drop_DH_featurename = ['datasize', 'max_key','minkey',
                               'pwl_num4', 'max_size4', 'min_size4', 'max_gap4', 'min_gap4',
                               'pwl_num8', 'max_size8', 'min_size8', 'max_gap8', 'min_gap8',
                               'pwl_num16', 'max_size16', 'min_size16', 'max_gap16', 'min_gap16',
                               'max_size32', 'min_size32', 'max_gap32', 'min_gap32',
                               'pwl_num64', 'max_size64', 'min_size64', 'max_gap64', 'min_gap64',
                               'pwl_num128', 'max_size128', 'min_size128', 'max_gap128', 'min_gap128',
                               'pwl_num256', 'max_size256', 'min_size256', 'max_gap256', 'min_gap256',
                               'pwl_num512', 'max_size512', 'min_size512', 'max_gap512', 'min_gap512',
                               'pwl_num1024', 'max_size1024', 'min_size1024', 'max_gap1024', 'min_gap1024',
                               'pwl_num2048', 'max_size2048', 'min_size2048', 'max_gap2048', 'min_gap2048',
                               'max_size4096', 'min_size4096', 'max_gap4096', 'min_gap4096']
        x_train = x_train.drop(drop_DH_featurename, axis=1)
        x_test = x_test.drop(drop_DH_featurename, axis=1)
        x_valid = x_valid.drop(drop_DH_featurename, axis=1)
    else:
        print('feature_type: ' , feature_type)


    # 验证 regret
    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valclfheader = generate_val_clf_header(label)
    valdata = total_valdata[valclfheader]

    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )
    intersected_df = clf_data_normalize(intersected_df)
    intersected_y = intersected_df['indexname'].copy(deep= True)
    # 求 intersectec_y 的unique labels
    intersect_label_values = intersected_df['indexname'].unique()
    for i_label in intersect_label_values:
        if (i_label not in class_name.keys() ):
            class_name[i_label] = len(class_name)
            num_class += 1
    for key,x in class_name.items():
        intersected_df.loc[intersected_df['indexname'] == key, 'indexname'] = x

   

    intersected_df[['indexname']] = intersected_df[['indexname']].astype('int')
    intersected_y_class_num = intersected_df['indexname']
    intersected_data_ops = intersected_df[['dataname','opsname']]
    intersected_x = intersected_df.drop(['dataname','opsname','indexname',label], axis=1)
    intersect_num = len(intersected_y)
    intersected_x = intersected_x[x_train.columns]

    data_num = len(x_train) + len(x_test) + len(x_valid)
    # print ('Lord, I needs You! we have loaded the clf data')

    class_file =  respath + 'class_xgboost4Clf_' + label + str(data_num) + '.json'
    class_json = json.dumps(class_name, ensure_ascii=False, default=default_dump)
    with open(class_file, 'w') as json_file:
        json_file.write(class_json)

    feature_name  = list(x_train.columns)
    feature_num = len(feature_name)
    print('*'*90)
    # print('feature_name: ', feature_name)
    print('class_name: ', num_class)
    print('the number of features is ', len(feature_name))
    # print('lable: ', label)
    print('train: ',len(x_train),len(y_train),'  test: ',len(x_test),len(y_test), '   valid: ',len(x_valid),len(y_valid))

    # _clf parameters
    params = {'learning_rate': 0.3, 'n_estimators': 1500, 'gamma': 0.0, 'max_depth': 6, 'min_child_weight': 1,
                    'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                    'seed': 9958, 'objective': 'multi:softmax','eval_metric':'merror','num_class':num_class}
    para_file = respath + 'params_xgboost4Clf_' + label + str(data_num) + '.json'
    if adapt_v:
        modelpath = respath + 'model_xgboost4Clf_' + label + str(data_num) + '_' + str(
            feature_num) + '_' + adapt_v + '.pkl'
    else:
        modelpath = respath + 'model_xgboost4Clf_' + label + str(data_num) + '_' + str(
            feature_num) + '.pkl'
    if (test_train == False and os.path.exists(modelpath)):
        clf_mod = joblib.load(modelpath)
        traintime = None
        if (os.path.exists(para_file)):
            with open(para_file, 'r') as f:
                params = json.load(f)
    else:
        if (os.path.exists(para_file)):
            with open(para_file, 'r') as f:
                params = json.load(f)
        else:
            # _clf parameter tuning
            params_space = {
                'n_estimators': hyperopt.hp.quniform("n_estimators", 100, 1200, 15),
                'max_depth': hyperopt.hp.choice("max_depth", np.linspace(1, 10, 10, dtype=int)),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
                'reg_lambda': 1,
                'subsample': 1,
                'reg_alpha': 0,
                'min_child_weight': hyperopt.hp.choice("min_child_weight", np.linspace(1, 10, 10, dtype=int)),
                'colsample_bytree': 1,
                'colsample_bylevel': 1,
                'gamma': 0,
                'num_class':num_class,
                'objective': 'multi:softmax',
                'eval_metric':'merror'
            }
            best = run_clf_xgb_hypertuning(params_space, x_train, y_train, x_valid, y_valid)
            params.update(best)

            params_json = json.dumps(params, ensure_ascii=False, default=default_dump)
            with open(para_file, 'w') as json_file:
                json_file.write(params_json)


        # _clf train
        start = time.perf_counter()

        clf_mod = xgb.XGBClassifier(objective='multi:softmax',
                                    max_depth=int(params['max_depth']),
                                    learning_rate=params['learning_rate'],
                                    subsample=params['subsample'],
                                    colsample_bytree=params['colsample_bytree'],
                                    n_estimators=int(params['n_estimators']),
                                    min_child_weight=int(params['min_child_weight']),
                                    num_class=num_class,
                                    eval_metric= 'merror',
                                    gpu_id = 0,
                                    )

        clf_mod.fit(x_train, y_train)
        end = time.perf_counter()
        traintime = end-start
        joblib.dump(clf_mod,modelpath)

    # compute regret
    start1 = time.perf_counter_ns()
    intersected_y_preds = clf_mod.predict(intersected_x)
    end1 = time.perf_counter_ns()
    probability = clf_mod.predict_proba(intersected_x)
    index_max = np.argmax(probability, axis=1)
    # print(index_max.shape)
    max_pro = probability[range(probability.shape[0]), index_max]
    max_c_80 = np.sum(max_pro>0.8)/intersect_num
    acc_score = clf_mod.score(intersected_x,intersected_y_class_num)
    intersected_y_preds = pd.Series(intersected_y_preds.tolist()).to_frame()
    intersected_y_preds.columns = ['indexname']
    value2class = {v : k for k, v in class_name.items()}
    for key,x in value2class.items():
        intersected_y_preds.loc[intersected_y_preds['indexname'] == key, 'indexname'] = x
    intersected_y_preds = pd.Series(intersected_y_preds['indexname'].values)
    inter_reference_time = end1 - start1


    valdata_accuracy = accuracy(intersected_y,intersected_y_preds)


    if adapt_v:
        clf_regret = None
        avg_regret = None
        not_find = None
    else:
        clf_regret,not_find  = compute_single_regret(label,intersected_data_ops,intersected_y_preds,intersected_y)

        avg_regret = clf_regret/(intersect_num-not_find)

    res_file = respath+'result_clf_xgboost'
    if adapt_v:
        res_file += '_adapter'
    res_dict = res_dict_write(label,val_data,modeltype,feature_type,sampling,feature_num,traintime,clf_mod,
                              adapt_v,data_num,0,valdata_accuracy,inter_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find,max_c_80,acc_score)

    print (res_dict)
    write_res(res_file,res_dict,'header')

    return modelpath,para_file,data_num


def constraint_XGB_Clf(objective,con_label,allsampling,feature_type,reg_data_num,v_data_num,sampling,val_data = 'finish',
            modeltype = 'clfXGB',adapt_v=None,test_train = False):
    print('Lord, please be with us, today is Jan 4')
    print('Lord, this is XGB for classification task for: ', objective, ' with constraint: ', con_label)
    labels = ['bulkloadtime','throughput','indexsize']
    # split train and test
                # load clf data

    clf_file = os.path.join(clf_datapath, objective+'_'+allsampling+str(reg_data_num))
    clfdata = pd.read_csv(clf_file)
    clfheader = generate_clf_header(objective)


    clfdata = add_new_feature_rq_i(clfdata)

    train_unique_class = [1,0]
    all_unique_class = [0,2]
    while (set(train_unique_class) != (set(all_unique_class))):
        x_train,y_train,x_test,y_test,x_valid,y_valid,num_class,class_name = \
            split_train_test(clfdata,clfheader,objective)
        train_unique_class = y_train.unique()
        all_unique_class = list(range(0,num_class))
        x_train = clf_data_normalize(x_train)
        x_test = clf_data_normalize(x_test)
        x_valid = clf_data_normalize(x_valid)
        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)




    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']]
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)


    #  regret
    val_file = os.path.join(val_datapath,objective + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valclfheader = generate_val_clf_header(objective)
    valdata = total_valdata[valclfheader]

    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )
    intersected_df = clf_data_normalize(intersected_df)
    intersected_y = intersected_df['indexname'].copy(deep= True)
    for key,x in class_name.items():
        intersected_df.loc[intersected_df['indexname'] == key, 'indexname'] = x

    intersected_df[['indexname']] = intersected_df[['indexname']].astype('int')
    intersected_y_class_num = intersected_df['indexname']
    intersected_data_ops = intersected_df[['dataname','opsname']]
    intersected_x = intersected_df.drop(['dataname','opsname','indexname',objective], axis=1)
    intersect_num = len(intersected_y)
    intersected_x = intersected_x[x_train.columns]

    data_num = len(x_train) + len(x_test) + len(x_valid)
    # print ('Lord, I needs You! we have loaded the clf data')

    class_file =  respath + 'class_xgboost4Clf_' + objective + str(data_num) + '.json'
    class_json = json.dumps(class_name, ensure_ascii=False, default=default_dump)
    with open(class_file, 'w') as json_file:
        json_file.write(class_json)

    feature_name  = list(x_train.columns)
    feature_num = len(feature_name)
    print('*'*90)
    # print('feature_name: ', feature_name)
    print('class_name: ', num_class)
    print('the number of features is ', len(feature_name))
    # print('lable: ', label)
    print('train: ',len(x_train),len(y_train),'  test: ',len(x_test),len(y_test), '   valid: ',len(x_valid),len(y_valid))

    # _clf parameters
    params = {'learning_rate': 0.3, 'n_estimators': 1500, 'gamma': 0.0, 'max_depth': 6, 'min_child_weight': 1,
              'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
              'seed': 9958, 'objective': 'multi:softmax','eval_metric':'merror','num_class':num_class}
    para_file = respath + 'params_xgboost4Clf_' + objective + str(data_num) + '.json'

    modelpath = respath + 'model_xgboost4Clf_' + objective + str(data_num) + '_' + str(
        feature_num) + '.pkl'
    if (test_train == False and os.path.exists(modelpath)):
        clf_mod = joblib.load(modelpath)
        traintime = None
        if (os.path.exists(para_file)):
            with open(para_file, 'r') as f:
                params = json.load(f)
    else:
        if (os.path.exists(para_file)):
            with open(para_file, 'r') as f:
                params = json.load(f)
        else:
            # _clf parameter tuning
            params_space = {
                'n_estimators': hyperopt.hp.quniform("n_estimators", 100, 1200, 15),
                'max_depth': hyperopt.hp.choice("max_depth", np.linspace(1, 10, 10, dtype=int)),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
                'reg_lambda': 1,
                'subsample': 1,
                'reg_alpha': 0,
                'min_child_weight': hyperopt.hp.choice("min_child_weight", np.linspace(1, 10, 10, dtype=int)),
                'colsample_bytree': 1,
                'colsample_bylevel': 1,
                'gamma': 0,
                'num_class':num_class,
                'objective': 'multi:softmax',
                'eval_metric':'merror'
            }
            best = run_clf_xgb_hypertuning(params_space, x_train, y_train, x_valid, y_valid)
            params.update(best)

            params_json = json.dumps(params, ensure_ascii=False, default=default_dump)
            with open(para_file, 'w') as json_file:
                json_file.write(params_json)


        # _clf train
        start = time.perf_counter()

        clf_mod = xgb.XGBClassifier(objective='multi:softmax',
                                    max_depth=int(params['max_depth']),
                                    learning_rate=params['learning_rate'],
                                    subsample=params['subsample'],
                                    colsample_bytree=params['colsample_bytree'],
                                    n_estimators=int(params['n_estimators']),
                                    min_child_weight=int(params['min_child_weight']),
                                    num_class=num_class,
                                    eval_metric= 'merror',
                                    gpu_id = 0,
                                    )

        clf_mod.fit(x_train, y_train)
        end = time.perf_counter()
        traintime = end-start
        joblib.dump(clf_mod,modelpath)

    # regret
    start1 = time.perf_counter_ns()
    intersected_y_preds = clf_mod.predict(intersected_x)
    end1 = time.perf_counter_ns()
    probability = clf_mod.predict_proba(intersected_x)
    index_max = np.argmax(probability, axis=1)
    # print(index_max.shape)
    max_pro = probability[range(probability.shape[0]), index_max]
    max_c_80 = np.sum(max_pro>0.8)/intersect_num
    acc_score = clf_mod.score(intersected_x,intersected_y_class_num)
    intersected_y_preds = pd.Series(intersected_y_preds.tolist()).to_frame()
    intersected_y_preds.columns = ['indexname']
    value2class = {v : k for k, v in class_name.items()}
    for key,x in value2class.items():
        intersected_y_preds.loc[intersected_y_preds['indexname'] == key, 'indexname'] = x
    intersected_y_preds = pd.Series(intersected_y_preds['indexname'].values)
    inter_reference_time = end1 - start1


    valdata_accuracy = accuracy(intersected_y,intersected_y_preds)


    c_seed = 9958521
    total_constraints,total_metric_min, total_metric_max = generate_groupby_constraint(labels,intersect_df=intersected_df,c_seed=c_seed)
    # total_constraints,total_metric_min, total_metric_max = generate_constraint(labels,c_num=intersect_num)
    # objective,constraints,metric_min,metric_max,

    clf_regret,not_find  = constraint_compute_regret(objective,con_label,total_constraints,intersected_data_ops,intersected_y_preds,intersected_y)

    avg_regret = clf_regret/(intersect_num-not_find)

    res_file = respath+'constraint_result_clf_xgboost'


    res_dict = constraint_res_dict_write(objective,val_data,modeltype,feature_type,sampling,feature_num,traintime,
                              adapt_v,data_num,0,valdata_accuracy,inter_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find,
                              con_label,total_constraints[0][con_label],c_seed)

    print (res_dict)
    write_res(res_file,res_dict,'header')

    return modelpath,para_file,data_num




# update XGB-clf：data adaptation, workload adaptation, index adaption
def incremental_XGB_Clf(label,modelpath,parapath,adapt_v,allsampling,feature_type,init_data_num,adapt_data_num,v_data_num,sampling,
                        val_data = 'finish',modeltype= 'clfXGB'):
    print('Lord, thank You! we test the adaptation of XGB-Clf')
    # load init model
    init_model = load_model(modelpath,label)

    if (os.path.exists(parapath)):
        with open(parapath,'r') as f:
            params = json.load(f)
    else:
        print ("Lord, please come! the params does not exist!")
        return

    # load init clf data
    initclf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(init_data_num)+'_'+adapt_v)
    initclfdata = pd.read_csv(initclf_file)
    initclfheader = generate_clf_header(label)
    # add is_i, is_rq, lookup
    initclfdata = add_new_feature_rq_i(initclfdata)
    init_x_train,init_y_train,init_x_test,init_y_test,init_x_valid,init_y_valid, \
    init_num_class,init_class_name = init_split_train_test(initclfdata,initclfheader,label)
    init_x_train = clf_data_normalize(init_x_train)
    init_x_test = clf_data_normalize(init_x_test)
    init_x_valid = clf_data_normalize(init_x_valid)

    init_x_all = pd.concat([init_x_test,init_x_valid])
    init_y_all = pd.concat([init_y_test,init_y_valid])
    init_x_all.reset_index(inplace=True, drop=True)
    init_y_all.reset_index(inplace=True, drop=True)
    init_x_all = init_x_all.drop(['dataname','opsname'], axis=1)

    # load adapt clf data
    clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(adapt_data_num)+'_'+adapt_v)
    clfdata = pd.read_csv(clf_file)
    clfheader = generate_clf_header(label)
    # 加入特征 is_i, is_rq, lookup
    clfdata = add_new_feature_rq_i(clfdata)
    adapt_label_values = clfdata['indexname'].unique()
    for i_label in adapt_label_values:
        if (i_label not in init_class_name.keys() ):
            init_class_name[i_label] = len(init_class_name)
            init_num_class += 1
    x_train,y_train,x_test,y_test,x_valid,y_valid,adapt_label_values,num_class = \
        adapt_split_train_test(clfdata,clfheader,label,init_class_name)
    x_train = clf_data_normalize(x_train)
    x_test = clf_data_normalize(x_test)
    x_valid = clf_data_normalize(x_valid)
    if num_class != init_num_class:
        # 在x_trian 中分别插入一条adapt class member 中没有的
        for k, v in init_class_name.items():
            if k not in adapt_label_values:
                k_data = init_x_train[ init_x_train['indexname']== v ]
                if len(k_data) == 0:
                    k_data = init_x_all[ init_x_all['indexname']== v ]
                k_y =  pd.Series(k_data['indexname'].iloc[0])
                k_data = k_data.drop(['indexname'], axis=1).iloc[0]
                x_train = x_train._append(k_data,ignore_index=True)
                y_train = y_train._append(k_y, ignore_index=True)
    init_x_train = init_x_train.drop(['indexname'], axis=1)
    init_x_valid = init_x_valid.drop(['indexname'], axis=1)
    init_x_test = init_x_test.drop(['indexname'], axis=1)
    init_x_all = init_x_all.drop(['indexname'], axis=1)
    x_all = pd.concat([x_test,x_valid])
    y_all = pd.concat([y_test,y_valid])
    x_all.reset_index(inplace=True, drop=True)
    y_all.reset_index(inplace=True, drop=True)
    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']]
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    init_x_train = init_x_train.drop(['dataname','opsname'], axis=1)
    init_x_valid = init_x_valid.drop(['dataname','opsname'], axis=1)
    init_x_test = init_x_test.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)
    # 验证 regret
    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valclfheader = generate_val_clf_header(label)
    valdata = total_valdata[valclfheader]

    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )
    intersected_df = clf_data_normalize(intersected_df)
    intersected_y = intersected_df['indexname'].copy(deep= True)
    intersect_label_values = intersected_df['indexname'].unique()
    for i_label in intersect_label_values:
        if (i_label not in init_class_name.keys() ):
            init_class_name[i_label] = len(init_class_name)
            init_num_class += 1
    for key,x in init_class_name.items():
        intersected_df.loc[intersected_df['indexname'] == key, 'indexname'] = x

    intersected_df[['indexname']] = intersected_df[['indexname']].astype('int')
    intersected_y_class_num = intersected_df['indexname']
    intersected_data_ops = intersected_df[['dataname','opsname']]
    intersected_x = intersected_df.drop(['dataname','opsname','indexname',label], axis=1)
    intersect_num = len(intersected_y)

    feature_name  = list(x_train.columns)
    feature_num = len(feature_name)

    init_x_num = len(init_x_train) + len(init_x_test) + len(init_x_valid)
    incremental_x_num = len(x_train) + len(x_test) + len(x_valid)
    res_file = respath+'clf_result_xgboost_adapter'

    # new data in init model 
    start1 = time.perf_counter_ns()
    intersected_y_preds = init_model.predict(intersected_x)
    end1 = time.perf_counter_ns()

    probability = init_model.predict_proba(intersected_x)
    index_max = np.argmax(probability, axis=1)
    # print(index_max.shape)
    max_pro = probability[range(probability.shape[0]), index_max]
    max_c_80 = np.sum(max_pro>0.8)/intersect_num
    acc_score = init_model.score(intersected_x,intersected_y_class_num)

    intersected_y_preds = pd.Series(intersected_y_preds.tolist()).to_frame()
    intersected_y_preds.columns = ['indexname']
    value2class = {v : k for k, v in init_class_name.items()}
    for key,x in value2class.items():
        intersected_y_preds.loc[intersected_y_preds['indexname'] == key, 'indexname'] = x
    intersected_y_preds = pd.Series(intersected_y_preds['indexname'].values)
    vald_reference_time = end1 - start1

    # validate 
    valdata_accuracy = accuracy(intersected_y,intersected_y_preds)

    clf_regret,not_find  = compute_single_regret(label,intersected_data_ops,intersected_y_preds,intersected_y)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(label,val_data,'init_'+modeltype,feature_type,sampling,feature_num,0,init_model,
                              adapt_v,init_x_num,incremental_x_num,valdata_accuracy,vald_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find,max_c_80,acc_score)


    print (res_dict)
    write_res(res_file,res_dict,'header')

    print ('Lord, I needs You, from ever to ever!')
    # 合并init_train，new_train
    merge_x_train = pd.concat([init_x_train,x_train,init_x_test,init_x_valid])
    merge_y_train = pd.concat([init_y_train,y_train,init_y_test,init_y_valid])
    merge_train = pd.concat([merge_x_train,merge_y_train],axis=1).reset_index()
    a = merge_train.columns
    merge_train.rename(columns={0:'indexname'},inplace = True)
    train_unique_class = [1,0]
    all_unique_class = [0,2]
    while (set(train_unique_class) != (set(all_unique_class))):
        all_x_train,all_y_train,all_x_test,all_y_test,all_x_valid,all_y_valid,num_class,class_name \
            = split_retrain_test(merge_train)
        train_unique_class = y_train.unique()
        all_unique_class = list(range(0,num_class))
    # incremental learning new data
    adapt_model = xgb.XGBClassifier(objective='multi:softmax',
                                    max_depth=int(params['max_depth']),
                                    learning_rate=params['learning_rate'],
                                    subsample=params['subsample'],
                                    colsample_bytree=params['colsample_bytree'],
                                    n_estimators=int(params['n_estimators']),
                                    min_child_weight=int(params['min_child_weight']),
                                    num_class=init_num_class,
                                    eval_metric= 'merror',
                                    )

    start = time.perf_counter()
    adapt_model.fit(all_x_train,all_y_train, xgb_model = init_model.get_booster())
    end = time.perf_counter()
    adapttime = end-start

    # new data in adapt model 
    start1 = time.perf_counter_ns()
    intersected_y_preds = adapt_model.predict(intersected_x)
    end1 = time.perf_counter_ns()
    probability = adapt_model.predict_proba(intersected_x)
    index_max = np.argmax(probability, axis=1)
    # print(index_max.shape)
    max_pro = probability[range(probability.shape[0]), index_max]
    max_c_80 = np.sum(max_pro>0.8)/intersect_num
    acc_score = adapt_model.score(intersected_x,intersected_y_class_num)

    intersected_y_preds = pd.Series(intersected_y_preds.tolist()).to_frame()
    intersected_y_preds.columns = ['indexname']
    value2class = {v : k for k, v in init_class_name.items()}
    for key,x in value2class.items():
        intersected_y_preds.loc[intersected_y_preds['indexname'] == key, 'indexname'] = x
    intersected_y_preds = pd.Series(intersected_y_preds['indexname'].values)
    vald_reference_time = end1 - start1

    # vallidate
    valdata_accuracy = accuracy(intersected_y,intersected_y_preds)

    clf_regret,not_find  = compute_single_regret(label,intersected_data_ops,intersected_y_preds,intersected_y)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(label,val_data,'adapt_'+modeltype,feature_type,sampling,feature_num,adapttime,adapt_model,
                              adapt_v,init_x_num,incremental_x_num,valdata_accuracy,vald_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find,max_c_80,acc_score)

    print (res_dict)
    write_res(res_file,res_dict)


    # 测试 retrain model
    retrain_model = xgb.XGBClassifier(objective='multi:softmax',
                                      max_depth=int(params['max_depth']),
                                      learning_rate=params['learning_rate'],
                                      subsample=params['subsample'],
                                      colsample_bytree=params['colsample_bytree'],
                                      n_estimators=int(params['n_estimators']),
                                      min_child_weight=int(params['min_child_weight']),
                                      num_class=init_num_class,
                                      eval_metric= 'merror',
                                      gpu_id = 0
                                      )

    start = time.perf_counter()
    retrain_model.fit(all_x_train,all_y_train)
    end = time.perf_counter()
    retraintime = end-start

    # new data in new model 
    start1 = time.perf_counter_ns()
    intersected_y_preds = retrain_model.predict(intersected_x)
    end1 = time.perf_counter_ns()
    probability = retrain_model.predict_proba(intersected_x)
    index_max = np.argmax(probability, axis=1)
    # print(index_max.shape)
    max_pro = probability[range(probability.shape[0]), index_max]
    max_c_80 = np.sum(max_pro>0.8)/intersect_num
    acc_score = retrain_model.score(intersected_x,intersected_y_class_num)

    intersected_y_preds = pd.Series(intersected_y_preds.tolist()).to_frame()
    intersected_y_preds.columns = ['indexname']
    value2class = {v : k for k, v in init_class_name.items()}
    for key,x in value2class.items():
        intersected_y_preds.loc[intersected_y_preds['indexname'] == key, 'indexname'] = x
    intersected_y_preds = pd.Series(intersected_y_preds['indexname'].values)
    vald_reference_time = end1 - start1


    valdata_accuracy = accuracy(intersected_y,intersected_y_preds)

    clf_regret,not_find  = compute_single_regret(label,intersected_data_ops,intersected_y_preds,intersected_y)
    avg_regret = clf_regret/(intersect_num-not_find)   

    res_dict = res_dict_write(label,val_data,'retrain_'+modeltype,feature_type,sampling,feature_num,retraintime,retrain_model,
                              adapt_v,init_x_num,incremental_x_num,valdata_accuracy,vald_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find,max_c_80,acc_score)

    print (res_dict)
    write_res(res_file,res_dict)



def CART_Clf(label,allsampling,feature_type,modeltype = 'clfXGB',adapter=None,test_train = True):
    print('Lord, thank You, this is for CART decision tree')

def RandomForest_Clf(label,allsampling,feature_type,reg_data_num,v_data_num,sampling,val_data = 'finish',
                     modeltype = 'clfRf',adapt_v=None,test_train = True):
    print('Lord, please be with us, today is Jan 17')
    print('Lord, this is Rnadom Forest for classification task: ', label)

    # split train and test
    # load clf data
    if adapt_v:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num)+'_init_'+adapt_v)
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)
    else:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num))
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)
    # and  is_i, is_rq, lookup
    clfdata = add_new_feature_rq_i(clfdata)

    train_unique_class = [1,2]
    all_unique_class = [0,1]
    while (set(train_unique_class) != (set(all_unique_class))):
        x_train,y_train,x_test,y_test,x_valid,y_valid,num_class,class_name = \
            split_train_test(clfdata,clfheader,label)
        x_train = clf_data_normalize(x_train)
        x_test = clf_data_normalize(x_test)
        x_valid = clf_data_normalize(x_valid)
        train_unique_class = y_train.unique()
        all_unique_class = list(range(0,num_class))
        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)



    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']]
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)


    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valclfheader = generate_val_clf_header(label)

    valdata = total_valdata[valclfheader]

    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )
    intersected_df = clf_data_normalize(intersected_df)
    intersected_y = intersected_df['indexname'].copy(deep= True)
    for key,x in class_name.items():
        intersected_df.loc[intersected_df['indexname'] == key, 'indexname'] = x

    intersected_df[['indexname']] = intersected_df[['indexname']].astype('int')
    intersected_y_class_num = intersected_df['indexname']
    intersected_data_ops = intersected_df[['dataname','opsname']]
    intersected_x = intersected_df.drop(['dataname','opsname','indexname',label], axis=1)
    intersect_num = len(intersected_y)

    data_num = len(x_train) + len(x_test) + len(x_valid)
    # print ('Lord, I needs You! we have loaded the clf data')

    class_file =  respath + 'class_RandomForest4Clf_' + label + str(data_num) + '.json'
    class_json = json.dumps(class_name, ensure_ascii=False, default=default_dump)
    with open(class_file, 'w') as json_file:
        json_file.write(class_json)

    feature_name  = list(x_train.columns)
    feature_num = len(feature_name)
    print('*'*90)
    # print('feature_name: ', feature_name)
    print('class_name: ', num_class)
    print('the number of features is ', len(feature_name))
    # print('lable: ', label)
    print('train: ',len(x_train),len(y_train),'  test: ',len(x_test),len(y_test), '   valid: ',len(x_valid),len(y_valid))

    # _clf parameters

    para_file = respath + 'params_RandomForest4Clf_' + label + str(data_num) + '.json'
    if adapt_v:
        modelpath = respath + 'model_RandomForest4Clf_' + label + str(data_num) + '_' + str(
            feature_num) + '_' + adapt_v + '.pkl'
    else:
        modelpath = respath + 'model_RandomForest4Clf_' + label + str(data_num) + '_' + str(
            feature_num) + '.pkl'
    if (test_train == False and os.path.exists(modelpath)):
        clf_mod = joblib.load(modelpath)
        traintime = None
        if (os.path.exists(para_file)):
            with open(para_file, 'r') as f:
                params_space = json.load(f)
    else:
        if (os.path.exists(para_file)):
            with open(para_file, 'r') as f:
                params_space = json.load(f)
        else:
            # _clf parameter tuning
            params_space = {
                'n_estimators': hyperopt.hp.quniform("n_estimators", 100, 1200, 15),
                'max_depth': hyperopt.hp.choice("max_depth", np.linspace(1, 10, 10, dtype=int)),
                'min_samples_split':hyperopt.hp.uniform('min_samples_split',2,6),
                'min_samples_leaf':hyperopt.hp.uniform('min_samples_leaf',1,5),
                'oob_score':True,
                'n_jobs': -1,
                'seed':9958,
                # 'criterion': hyperopt.hp.choice('criterion', ['entropy', 'gini'])
            }
            best = run_clf_RF_hypertuning(params_space, x_train, y_train, x_valid, y_valid)
            params_space.update(best)

            params_json = json.dumps(params_space, ensure_ascii=False, default=default_dump)
            with open(para_file, 'w') as json_file:
                json_file.write(params_json)


        # _clf train
        start = time.perf_counter()

        clf_mod = RandomForestClassifier( max_depth=int(params_space['max_depth']),
                                          # criterion=params_space['criterion'],
                                          min_samples_leaf=int(params_space['min_samples_leaf']),
                                         min_samples_split=int(params_space['min_samples_split']),
                                         n_estimators=int(params_space['n_estimators']),
                                         n_jobs= params_space['n_jobs'],
                                         warm_start=True
                                    )

        clf_mod.fit(x_train, y_train)
        end = time.perf_counter()
        traintime = end-start
        joblib.dump(clf_mod,modelpath)


    # regret
    start1 = time.perf_counter_ns()
    intersected_y_preds = clf_mod.predict(intersected_x)
    end1 = time.perf_counter_ns()
    probability = clf_mod.predict_proba(intersected_x)
    index_max = np.argmax(probability, axis=1)
    # print(index_max.shape)
    max_pro = probability[range(probability.shape[0]), index_max]
    max_c_80 = np.sum(max_pro>0.8)/intersect_num
    acc_score = clf_mod.score(intersected_x,intersected_y_class_num)
    intersected_y_preds = pd.Series(intersected_y_preds.tolist()).to_frame()
    intersected_y_preds.columns = ['indexname']
    value2class = {v : k for k, v in class_name.items()}
    for key,x in value2class.items():
        intersected_y_preds.loc[intersected_y_preds['indexname'] == key, 'indexname'] = x
    intersected_y_preds = pd.Series(intersected_y_preds['indexname'].values)
    inter_reference_time = end1 - start1


    valdata_accuracy = accuracy(intersected_y,intersected_y_preds)


    if adapt_v:
        clf_regret = None
        avg_regret = None
        not_find = None
    else:
        clf_regret,not_find  = compute_single_regret(label,intersected_data_ops,intersected_y_preds,intersected_y)

        avg_regret = clf_regret/(intersect_num-not_find)


    res_file = respath+'result_clf_xgboost'
    res_dict = res_dict_write(label,val_data,modeltype,feature_type,sampling,feature_num,traintime,clf_mod,
                              adapt_v,data_num,0,valdata_accuracy,inter_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find,max_c_80,acc_score)

    print (res_dict)
    write_res(res_file,res_dict,'header')

    return modelpath,para_file,data_num
# tanks Lord, success
def run_XGB_clf():
    feature_type = "hybridencoder"

    sampling = 'systematic_10thousand'
    # sampling = 'uniform_10thousand'
    # sampling = 'stratified_10thousand'
    pos = sampling.find('_')
    samplinglist = list(sampling)
    samplinglist.insert(pos, '_sampling')
    allsampling = ''.join(samplinglist)
    embeding_file = '/home/xxx/index_selection/publish_data/total_feature_' + allsampling

    # single objective 已测试通， 感谢天主
    reg_data_num = 28008
    v_data_num = 29851
    tasks = ['bulkloadtime','throughput','indexsize']
    for label in tasks:
        XGB_Clf(label,allsampling,feature_type,reg_data_num,v_data_num,sampling)  # all data

    print('Lord, this is classification alternatives')


    print('Lord, this is classification alternatives')



def run_performance_constraint():
    print('Lord, thank You, thank be with me~ this is performance constraint')
    feature_type = "hybridencoder"
    sampling = 'systematic_10thousand'
    # sampling = 'uniform_10thousand'
    # sampling = 'stratified_10thousand'
    pos = sampling.find('_')
    samplinglist = list(sampling)
    samplinglist.insert(pos, '_sampling')
    allsampling = ''.join(samplinglist)

    feature_num = 140
    reg_data_num = 28008
    v_data_num = 29851
    labels = ['bulkloadtime','throughput','indexsize']

    objective = 'throughput'
    con_label = 'bulkloadtime'
    for i in range (3):
        constraint_XGB_Clf(objective,con_label,allsampling,feature_type,reg_data_num,v_data_num,sampling)

    objective = 'throughput'
    con_label = 'indexsize'
    for i in range (3):
        constraint_XGB_Clf(objective,con_label,allsampling,feature_type,reg_data_num,v_data_num,sampling)


    objective = 'indexsize'
    con_label = 'throughput'
    for i in range (3):
        constraint_XGB_Clf(objective,con_label,allsampling,feature_type,reg_data_num,v_data_num,sampling)


    objective = 'indexsize'
    con_label = 'bulkloadtime'
    for i in range (3):
        constraint_XGB_Clf(objective,con_label,allsampling,feature_type,reg_data_num,v_data_num,sampling)


    objective = 'bulkloadtime'
    con_label = 'indexsize'
    for i in range (3):
        constraint_XGB_Clf(objective,con_label,allsampling,feature_type,reg_data_num,v_data_num,sampling)


    objective = 'bulkloadtime'
    con_label = 'throughput'
    for i in range (3):
        constraint_XGB_Clf(objective,con_label,allsampling,feature_type,reg_data_num,v_data_num,sampling)


def run_RandomForest_clf():
    feature_type = "hybridencoder"
    sampling = 'systematic_10thousand'
    # sampling = 'uniform_10thousand'
    # sampling = 'stratified_10thousand'
    pos = sampling.find('_')
    samplinglist = list(sampling)
    samplinglist.insert(pos, '_sampling')
    allsampling = ''.join(samplinglist)
    embeding_file = '/home/xxx/index_selection/publish_data/total_feature_' + allsampling

    # single objective 
    reg_data_num = 28008
    v_data_num = 29851
    tasks = ['bulkloadtime','throughput','indexsize']
    for label in tasks:
        RandomForest_Clf(label,allsampling,feature_type,reg_data_num,v_data_num,sampling)  # all data

    print('Lord, this is classification alternatives')





if __name__ == '__main__':

    for i in range(1):
        run_XGB_clf()  # all data ,

    # run_performance_constraint()   

    # for i in range(10):
    #     run_RandomForest_clf() 






