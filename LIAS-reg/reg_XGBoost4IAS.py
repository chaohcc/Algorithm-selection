## 2023/12/29
# this code is writen as the random forest and XGBBoost
# finall version

import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import hyperopt


from sklearn.ensemble import RandomForestRegressor
import os
import csv
import json,joblib
import time

import numpy as np
from util import evaluate_model,write_res,default_dump, \
    return_index,evaluate_result,generate_clf_header
from util import generate_index_header,index_fead_feature,get_clf_candidate
from util import run_hypertuning,run_RF_hypertuning
from feature_processor import FeatureProcessor, build_dataset, \
    load_dataset_config,build_adapt_dataset, build_init_dataset,build_retrain_dataset,new_init_dataset,new_adapt_dataset
from feature_processor import build_naive_dataset
from feature_processor import build_hybrid_dataset, build_hybrid_init_dataset,build_hybrid_adapt_dataset
from feature_processor import build_hybrid_sample_r_dataset,build_sample_r_dataset,\
    build_hybrid_adapt_sample_r_dataset,build_hybrid_init_sample_r_dataset,\
    build_init_sample_r_dataset,candidates_process,build_original_value_hybrid_sample_r_dataset
import xgboost as xgb
from LIB4regression import seed_everything
from sklearn.feature_selection import SelectFromModel

respath = './result/Regret_RandomForest_XGB/'
val_datapath = '/home/wamdm/chaohong/index_selection/validation_data/'



def single_prediction(reg_mod,candidates):
    # print("Lord, we begin to select the optimal index algorithm")
    # load models

    # model prediction
    pred = reg_mod.predict(candidates)
    return pred

def single_return_index(pred,feature2index,label,index_candidates,index_header):

    if label == 'throughput': # 找最大的pred
        # print('my Lord, thank You. Long is coming! this is ', label)
        idx = np.argmax(pred)
        # tuple = candidates.loc[idx]
        tuple = index_candidates.loc[idx]
        index_feature = tuple.loc[index_header]
        # index_feature2 = tuple2.loc[index_header]
        index_feature = [str(int(x)) for x in index_feature]
        index_feature = "".join(index_feature)
        index_name = feature2index[index_feature]
        # print('my Lord, the selected index is: ', index_name)
    # 如何验证 效果？
    elif label == 'bulkloadtime' or label == 'indexsize' : # 找最小的
        # print('my Lord, Long and chao need You! this is ', label)
        idx = np.argmin(pred)
        # tuple = candidates.loc[idx]
        tuple = index_candidates.loc[idx]
        index_feature = tuple.loc[index_header]
        # index_feature2 = tuple2.loc[index_header]
        index_feature = [str(int(x)) for x in index_feature]
        index_feature = "".join(index_feature)
        index_name = feature2index[index_feature]
        # print('my Lord, the selected index is: ', index_name)
    else:
        print('weighted label ~~ my Lord, thank You. i miss Long, please help him.')
        idx = np.argmax(pred)
        # tuple = candidates.loc[idx]
        tuple = index_candidates.loc[idx]
        index_feature = tuple.loc[index_header]
        # index_feature2 = tuple.loc[index_header]
        index_feature = [str(int(x)) for x in index_feature]
        index_feature = "".join(index_feature)
        index_name = feature2index[index_feature]
        # print('my Lord, the selected index is: ', index_name)

    return pred[idx],index_name

original_val_datapath = '/home/wamdm/chaohong/index_selection/original_validation_data/'
def single_compute_regret(label,best_x, select_x,dataname,opsname, allsampling = 'systematic_sampling_10thousand'):
    not_find = 0
    val_name = dataname+'_'+opsname.split('_')[0]+allsampling
    valpath = os.path.join(val_datapath,label)
    valfile = os.path.join(valpath,val_name)
    data = pd.read_csv(valfile)
    # if select_x not in ['ART','FAST','ARTOLC','FINEdex','LIPP','ALEX']:
    #     print(select_x)
    if select_x == best_x:
        regret = 0
    else:
        indexdata = data['indexname'].tolist()
        if (select_x in indexdata ):
            # rank = indexdata.index(select_x)
            # regret = rank / 13
            # if select_x == 'HOT':
            #     print (dataname, ',',opsname,',',select_x, ',',best_x)
            # if rank >3:
            rank = indexdata.index(select_x)
            # print (dataname, ',',opsname,',',select_x, ',',best_x,' ', rank)
            x_value = data[label].iloc[rank]
            regret = (1-(min(data[label].iloc[0],x_value)/max(data[label].iloc[0],x_value)))

        else:
            regret = 0
            not_find = 1
            # print (opsname)
            # print(select_x)
            # print (dataname, ',',opsname,',',select_x)
            # print(select_x)
        # pred 在第几位
    return regret,not_find

def index_single_compute_regret(label,select_x,dataname,opsname, allsampling = 'systematic_sampling_10thousand'):
    not_find = 0
    val_name = dataname+'_'+opsname.split('_')[0]+allsampling
    valpath = os.path.join(val_datapath,label)
    valfile = os.path.join(valpath,val_name)
    data = pd.read_csv(valfile)

    indexdata = data['indexname'].tolist()
    best_x = indexdata[0]
    if select_x == best_x:
        regret = 0
    else:
        if (select_x in indexdata ):
            # rank = indexdata.index(select_x)
            # regret = rank / 13
            # if select_x == 'HOT':
            #     print (dataname, ',',opsname,',',select_x, ',',best_x)
            # if rank >3:
            rank = indexdata.index(select_x)
            # print (dataname, ',',opsname,',',select_x, ',',best_x,' ', rank)
            x_value = data[label].iloc[rank]
            regret = (1-(min(data[label].iloc[0],x_value)/max(data[label].iloc[0],x_value)))
            if (regret> 1):
                print('Lord, chao needs You!')
        else:
            regret = 0
            not_find = 1
            # print (opsname)
            # print(select_x)
            # print (dataname, ',',opsname,',',select_x)
            # print(select_x)
        # pred 在第几位
    return regret,not_find


def index_regret_single_objeectie_Selector(label,reg_mod,valdata,feature_header,selection  = None):

    total_regret = 0
    total_not_find = 0
    acc_cnt = 0
    not_find_data = []
    index_library,feature2index =  index_fead_feature()
    index_header = generate_index_header()
    row_header = valdata.columns.tolist() + index_header
    num = len(valdata)
    for x in ['dataname','opsname',label]:
        row_header.remove(x)
    for i, row in valdata.iterrows():
        dataname = row['dataname']
        opsname = row['opsname']
        if ('wiki' in row['dataname'] or 'lognormal' in row['dataname']):
            is_duplicate = True
        else:
            is_duplicate = False
        if '_1m_' in dataname:
            is_delta = True
        else:
            is_delta = False

        row = row.drop(['dataname','opsname',label])
        insert_ratio = row['i']
        thread_num = row['thread']
        is_rq = row['is_rq']
        # if thread_num > 1:
        #     print('Lord, thank You, this is chao')
        candidates,index_candidates,available_index = \
            get_clf_candidate(row, is_duplicate,insert_ratio,thread_num,is_delta,is_rq,index_library,row_header)
        # candidates 归一化
        # candidates = candidates_process(candidates)
        candidates = candidates[feature_header]
        if (selection):
            candidates = selection.transform(candidates)
        # 根据modeltype 和 label 加载model
        pred = single_prediction(reg_mod,candidates)

        x_metric,select_x = single_return_index(pred,feature2index,label,index_candidates,index_header)
        # 计算regret
        regret, not_find = index_single_compute_regret(label, select_x,dataname,opsname)
        if not_find:
            valdata = valdata.drop(i)
        total_regret += regret
        total_not_find += not_find
        if (regret == 0 ):
            acc_cnt += 1
    all_accuracy = (acc_cnt-total_not_find) / (num-total_not_find)
    print('Lord, chao need You! please come! ')
    return total_regret, total_not_find,all_accuracy,valdata


    # evaluate_result('weighted',data_name,best_index,workload_embeding,x_metric)

def adapt_index_regret_single_objeectie_Selector(label,reg_mod,valdata,feature_header,selection  = None):

    total_regret = 0
    total_not_find = 0
    acc_cnt = 0
    not_find_data = []
    index_library,feature2index =  index_fead_feature()
    index_header = generate_index_header()
    row_header = valdata.columns.tolist() + index_header
    num = len(valdata)
    for x in ['dataname','opsname',label]:
        row_header.remove(x)
    for i, row in valdata.iterrows():
        dataname = row['dataname']
        opsname = row['opsname']
        if ('wiki' in row['dataname'] or 'lognormal' in row['dataname']):
            is_duplicate = True
        else:
            is_duplicate = False
        if '_1m_' in dataname:
            is_delta = True
        else:
            is_delta = False

        row = row.drop(['dataname','opsname',label])
        insert_ratio = row['i']
        thread_num = row['thread']
        is_rq = row['is_rq']
        # if thread_num > 1:
        #     print('Lord, thank You, this is chao')
        candidates,index_candidates,available_index = \
            get_clf_candidate(row, is_duplicate,insert_ratio,thread_num,is_delta,is_rq,index_library,row_header)
        # candidates 归一化
        # candidates = candidates_process(candidates)
        candidates = candidates[feature_header]
        if (selection):
            candidates = selection.transform(candidates)
        # 根据modeltype 和 label 加载model
        pred = adapt_single_prediction(reg_mod,candidates)

        x_metric,select_x = single_return_index(pred,feature2index,label,index_candidates,index_header)
        # 计算regret
        regret, not_find = index_single_compute_regret(label, select_x,dataname,opsname)
        if not_find:
            valdata = valdata.drop(i)
        total_regret += regret
        total_not_find += not_find
        if (regret == 0 ):
            acc_cnt += 1
    all_accuracy = (acc_cnt-total_not_find) / (num-total_not_find)
    print('Lord, chao need You! please come! ')
    return total_regret, total_not_find,all_accuracy,valdata


    # evaluate_result('weighted',data_name,best_index,workload_embeding,x_metric)



def regret_single_objeectie_Selector(label,reg_mod,valdata,feature_header,selection  = None):

    total_regret = 0
    total_not_find = 0
    acc_cnt = 0
    index_library,feature2index =  index_fead_feature()
    index_header = generate_index_header()
    data_ops_all = valdata[['dataname','opsname']]
    data_ops_all.insert(data_ops_all.shape[1],'indexname','')
    for i, row in valdata.iterrows():
        dataname = row['dataname']
        opsname = row['opsname']
        if ('wiki' in row['dataname'] or 'lognormal' in row['dataname']):
            is_duplicate = True
        else:
            is_duplicate = False
        if '_1m_' in dataname:
            is_delta = True
        else:
            is_delta = False
        best_x = row['indexname']
        row = row.drop(['dataname','opsname','indexname',label])
        insert_ratio = row['i']
        thread_num = row['thread']
        is_rq = row['is_rq']
        # if thread_num > 1:
        #     print('Lord, thank You, this is chao')
        candidates,index_candidates,available_index = \
            get_clf_candidate(row, is_duplicate,insert_ratio,thread_num,is_delta,is_rq,index_library)
        # candidates 归一化
        candidates = candidates_process(candidates)
        candidates = candidates[feature_header]
        if (selection):
            candidates = selection.transform(candidates)
        # 根据modeltype 和 label 加载model
        pred = single_prediction(reg_mod,candidates)

        x_metric,select_x = single_return_index(pred,feature2index,label,index_candidates,index_header)
        # 计算regret
        regret, not_find = single_compute_regret(label,best_x, select_x,dataname,opsname)
        total_regret += regret
        total_not_find += not_find
        if (regret == 0 ):
            acc_cnt += 1
    all_accuracy = (acc_cnt-total_not_find) / (len(valdata)-total_not_find)
    print('Lord, chao need You! please come! ')
    return total_regret, total_not_find,all_accuracy


    # evaluate_result('weighted',data_name,best_index,workload_embeding,x_metric)


def res_dict_write(res_dict,label,val_data,modeltype,feature_type,sampling,feature_num,trainingtime,clf_model,
                   adapt_v,init_x_num,incremental_x_num,valdata_accuracy,vald_reference_time,
                   clf_regret,intersect_num,avg_regret,not_found):

    res_dict['task'] = label
    res_dict['data'] = val_data
    res_dict['model'] = modeltype
    res_dict['feature'] = feature_type
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = trainingtime
    res_dict['n_estimators'] = clf_model.n_estimators
    if modeltype == 'singleRF':
        res_dict['learning_rate'] = None
        res_dict['min_child_weight'] = None
    else:
        res_dict['learning_rate'] = clf_model.learning_rate
        res_dict['min_child_weight'] = clf_model.min_child_weight
    res_dict['max_depth'] = clf_model.max_depth

    res_dict['adapter'] = adapt_v
    res_dict['init_data_num'] = init_x_num
    res_dict['adapt_data_num'] = incremental_x_num
    res_dict['accuracy'] = valdata_accuracy
    res_dict['reference_time'] = vald_reference_time
    res_dict['regret'] = clf_regret
    res_dict['intersect_num'] = intersect_num
    res_dict['avg_regret'] = avg_regret
    res_dict['not_found'] = not_found

    return res_dict


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



clf_datapath = '/home/wamdm/chaohong/index_selection/classification_data/'
# 应用hyperopt 进行调参
def Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,
                     feature_type = 'hybridencoder',modeltype = 'singleXGB',
                     adapter=None,test_train = True,val_data = 'finish'):


    if adapter:
        config_file = 'adapt_'+adapter+'_dataset_config.yaml'
    else:
        config_file = 'hasops_lias_dataset_config.yaml'

    args = {'config': './config/','configfile':'res_model_config.yaml',
            'dataset_id' : 'res_mtl',
            'datasetconfigfile':config_file}
    data_params = load_dataset_config(args['config'], args['dataset_id'],args['datasetconfigfile'])
    seed_everything(data_params['seed'])

    feature_encoder = FeatureProcessor(**data_params)

    # 加载 embedding
    # if label not in ['indexsize','bulkloadtime']:
    if label not in []:
        embedings = pd.read_csv(embeding_file)
        embed_head = []
        for i in range(32):
            embed_head.append('em'+str(i))
        embedings = embedings[embed_head]
    if 'stratified_' not in sampling:
        ratios = pd.read_csv(ratio_file)
        ratio_head = ['i','rq']
        ratios = ratios[ratio_head]
    # 把embeding 和 manual feature 合并
    if adapter:
        # if label not in ['indexsize','bulkloadtime']:
        if label not in []:
            if 'stratified_' in sampling:
                x_train,y_train,x_test,y_test,x_valid,y_valid = build_hybrid_init_dataset(feature_encoder, label,embedings,
                                                                                      **data_params)
            else:

                x_train,y_train,x_test,y_test,x_valid,y_valid = \
                    build_hybrid_init_sample_r_dataset(feature_encoder, label,embedings,
                                                   ratios,**data_params)
        else:
            if 'stratified_' in sampling:
                x_train,y_train,x_test,y_test,x_valid,y_valid = \
                    build_init_dataset(feature_encoder, label,**data_params)
            else:
                x_train,y_train,x_test,y_test,x_valid,y_valid = \
                    build_init_sample_r_dataset(feature_encoder, label,**data_params)
        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)
    else:
        # if label not in ["indexsize", "bulkloadtime"]:
        if label not in []:


            if feature_type == "naiveWorkloadFeature":
                if 'stratified_' in sampling:
                    x_train,y_train,x_test,y_test,x_valid,y_valid = build_dataset(feature_encoder, label,**data_params)
                else:
                    x_train,y_train,x_test,y_test,x_valid,y_valid = build_sample_r_dataset(feature_encoder, label,ratios,**data_params)

            else:
                if 'stratified_' in sampling:
                    x_train,y_train,x_test,y_test,x_valid,y_valid= build_hybrid_dataset(feature_encoder, label,embedings,
                                                                                        **data_params)
                else:
                    x_train,y_train,x_test,y_test,x_valid,y_valid= build_hybrid_sample_r_dataset(feature_encoder, label,embedings,ratios,
                                                                                    **data_params)

        else:
            # x_train,y_train,x_test,y_test,x_valid,y_valid = build_dataset(feature_encoder, label,**data_params)
            x_train,y_train,x_test,y_test,x_valid,y_valid = build_sample_r_dataset(feature_encoder, label,ratios,**data_params)
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
        elif feature_type == "index_f_naive":
            drop_x_fearture = generate_index_header()
            drop_x_fearture = drop_x_fearture[1:]
            x_train = x_train.drop(drop_x_fearture, axis=1)
            x_test = x_test.drop(drop_x_fearture, axis=1)
            x_valid = x_valid.drop(drop_x_fearture, axis=1)
            print('Lord, chao needs You!')
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


            print('Lord, chao needs You!')
        else:
            print('feature_type: ' , feature_type)

        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)

    data_num = len(x_train) + len(x_test) + len(x_valid)

    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']].copy(deep = True)
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)
    all_ddf = all_ddf.drop_duplicates(subset=['dataname','opsname'])
    # 验证 regret
    clfheader = generate_clf_header(label)
    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valdata = total_valdata[clfheader]
    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )

    # 加载一组分类数据， 划分train 和test, 与 intersected_df 求交集
    if adapter:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num)+'_'+adapter)
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)
    else:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num))
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)


    # clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name =\
    #     split_train_test(clfdata,clfheader,label,self_seed=9958)
    clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name = \
        split_train_test(clfdata,clfheader,label)
    clf_x_all = pd.concat([clf_x_test,clf_x_valid])
    clf_y_all = pd.concat([clf_y_test,clf_y_valid])
    clf_x_all.reset_index(inplace=True, drop=True)
    clf_y_all.reset_index(inplace=True, drop=True)

    clf_all_ddf = pd.concat([clf_x_all,clf_y_all],axis= 1 )
    clf_all_ddf =  clf_all_ddf[['dataname','opsname']]
    intersected_df = pd.merge(intersected_df,clf_all_ddf,on=['dataname','opsname'] )
    intersected_y = intersected_df['indexname'].copy(deep= True)

    intersect_num = len(intersected_y)

    feature_name  = list(x_train.columns)
    print('*'*90)
    print('the number of features is ', len(feature_name))
    print('lable: ', label)
    print('train: ',len(x_train), ' test: ',len(x_test),' valid: ',len(x_valid) )


    params_space = {'learning_rate': 0.3, 'n_estimators': 635, 'gamma': 0.0, 'max_depth': 8, 'min_child_weight': 2,
                    'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                    'seed': 9958,'objective':'reg:squarederror'}

    feature_num = len(feature_name)
    # all the sampling share the same parameters
    para_file = respath + 'params_hybrid_xgboost4regression_' + label + str(data_num) +sampling+ '.json'
    if adapter:
        para_file = respath + 'params_hybrid_xgboost4regression_' + label + str(28008) +sampling+ '.json'

    # each sampling has its own parameters
    # para_file = respath + 'params_hybrid_xgboost4regression_' + label + str(data_num) +'_' +sampling + '.json'
    if adapter:
        modelpath = respath+'model_hybrid_xgboost4regression_' + label + str(data_num) +'_'+str(feature_num)+'_'+sampling+'_'+adapter+'.pkl'
    else:
        modelpath = respath+'model_hybrid_xgboost4regression_' + label + str(data_num) +'_'+str(feature_num)+'_'+sampling+'.pkl'
    if (test_train==False and os.path.exists(modelpath)):
        reg_mod = joblib.load(modelpath)
        traintime = None
        if (os.path.exists(para_file)):
            with open(para_file,'r') as f:
                params_space = json.load(f)
    else:
        if (os.path.exists(para_file)):
            with open(para_file,'r') as f:
                params_space = json.load(f)
        else:
            # reg_mod = xgb.XGBRegressor(
            #     objective='reg:squarederror',
            #     max_depth=int(params_space['max_depth']),
            #     learning_rate=params_space['learning_rate'],
            #     subsample=params_space['subsample'],
            #     colsample_bytree=params_space['colsample_bytree'],
            #     n_estimators=int(params_space['n_estimators']),
            #     min_child_weight=int(params_space['min_child_weight']),
            #     tree_method='hist',
            #     gpu_id = '0'
            # )
            # reg_mod.fit(x_train, y_train, eval_metric='rmse', verbose=False)
            # selection = SelectFromModel(reg_mod, prefit=True,threshold=-np.inf,max_features=50)
            # print('特征选择阈值：', selection.threshold)
            # print('特征是否保留', selection.get_support())
            # choose_flag = selection.get_support()
            # idx = np.where(choose_flag==True)[0]
            # select_features = [feature_name[i] for i in idx]
            # params_space['select_feature'] = select_features
            # x_train = selection.transform(x_train)
            # x_test = selection.transform(x_test)
            # x_valid = selection.transform(x_valid)
            # print('selected features: ', select_features)
            # feature_num = len(params_space['select_feature'])

            params_space = {
                'n_estimators':hyperopt.hp.quniform("n_estimators",100,1500,15),
                'max_depth': hyperopt.hp.choice("max_depth",np.linspace(1, 10, 10, dtype=int)),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
                'reg_lambda':1,
                'subsample': 1,
                'reg_alpha': 0,
                'min_child_weight': hyperopt.hp.choice("min_child_weight",np.linspace(1, 10, 10, dtype=int)),
                'colsample_bytree': 1,
                'colsample_bylevel': 1,
                'gamma': 0,
                'tree_method': 'hist',
                'device': "cuda:0"

            }
            best = run_hypertuning(params_space,x_train,y_train,x_all,y_all)
            params_space.update(best)

            params_json = json.dumps(params_space,ensure_ascii=False, default=default_dump)
            with open(para_file,'w') as json_file:
                            json_file.write(params_json)
        if label in []:  # 'bulkloadtime'
            reg_mod = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=int(params_space['max_depth']),
                learning_rate=params_space['learning_rate'],
                subsample=params_space['subsample'],
                colsample_bytree=params_space['colsample_bytree'],
                n_estimators=int(params_space['n_estimators']),
                min_child_weight=int(params_space['min_child_weight']),
                tree_method='hist',
                # gpu_id = '0'
            )
            reg_mod.fit(x_train, y_train, eval_metric='rmse', verbose=False)
            selection = SelectFromModel(reg_mod, prefit=True,threshold=-np.inf,max_features=100)
            print('特征选择阈值：', selection.threshold)
            print('特征是否保留', selection.get_support())
            choose_flag = selection.get_support()
            idx = np.where(choose_flag==True)[0]
            select_features = [feature_name[i] for i in idx]
            params_space['select_feature'] = select_features
            x_train = selection.transform(x_train)
            x_all = selection.transform(x_all)
            x_test = selection.transform(x_test)
            x_valid = selection.transform(x_valid)
            print('selected features: ', select_features)
            feature_num = len(params_space['select_feature'])
            reg_mod = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=int(params_space['max_depth']),
                learning_rate=params_space['learning_rate'],
                subsample=params_space['subsample'],
                colsample_bytree=params_space['colsample_bytree'],
                n_estimators=int(params_space['n_estimators']),
                min_child_weight=int(params_space['min_child_weight']),
                # gpu_id = '0'
            )
            start = time.perf_counter()
            reg_mod.fit(x_train,y_train)
            end = time.perf_counter()
            traintime = end-start
            joblib.dump(reg_mod,modelpath)


            start = time.perf_counter_ns()
            y_pred_all = reg_mod.predict(x_all)
            end = time.perf_counter_ns()
            all_reference_time = end-start
            res_dict = evaluate_model(y_all,y_pred_all)

            # 应用模型进行回归, 并预测best_index
            res_file = respath+'result_xgboost4regression'
            if adapter:
                res_file += '_adapter'

            clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,reg_mod,intersected_df,selection)
            avg_regret = clf_regret/(intersect_num-not_find)

            res_dict = res_dict_write(res_dict,label,val_data,modeltype,feature_type,sampling,feature_num,traintime,reg_mod,
                                      adapter,data_num,0,valdata_accuracy,all_reference_time,
                                      clf_regret,intersect_num,avg_regret,not_find)

            print (res_dict)
            write_res(res_file,res_dict,'header')

            return modelpath,para_file,data_num


        reg_mod = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=int(params_space['max_depth']),
            learning_rate=params_space['learning_rate'],
            subsample=params_space['subsample'],
            colsample_bytree=params_space['colsample_bytree'],
            n_estimators=int(params_space['n_estimators']),
            min_child_weight=int(params_space['min_child_weight']),
            # gpu_id = '0'
        )
        start = time.perf_counter()
        reg_mod.fit(x_train,y_train)
        end = time.perf_counter()
        traintime = end-start
        joblib.dump(reg_mod,modelpath)


    start = time.perf_counter_ns()
    y_pred_all = reg_mod.predict(x_all)
    end = time.perf_counter_ns()
    all_reference_time = end-start
    res_dict = evaluate_model(y_all,y_pred_all)

    # 应用模型进行回归, 并预测best_index
    res_file = respath+'result_xgboost4regression'
    if adapter:
        res_file += '_adapter'

    clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,reg_mod,intersected_df,x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,val_data,modeltype,feature_type,sampling,feature_num,traintime,reg_mod,
                              adapter,data_num,0,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)

    print (res_dict)
    write_res(res_file,res_dict,'header')

    return modelpath,para_file,data_num

def init_Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,init_num,
                     feature_type = 'hybridencoder',modeltype = 'singleXGB',
                     adapter=None,test_train = True,val_data = 'finish'):


    seed_everything(9958)
    path = '/home/wamdm/chaohong/index_selection/total_init_adapt/'
    datafile = path+'normal_systematic_10thousand_total_feature_'+str(init_num)+'_'+adapter

    x_train,y_train,x_test,y_test,x_valid,y_valid = new_init_dataset(label,datafile)
    x_all = pd.concat([x_test,x_valid])
    y_all = pd.concat([y_test,y_valid])
    x_all.reset_index(inplace=True, drop=True)
    y_all.reset_index(inplace=True, drop=True)

    data_num = len(x_train) + len(x_test) + len(x_valid)

    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']].copy(deep = True)
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)
    all_ddf = all_ddf.drop_duplicates(subset=['dataname','opsname'])
    # 验证 regret
    clfheader = generate_clf_header(label)
    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valdata = total_valdata[clfheader]
    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )

    # 加载一组分类数据， 划分train 和test, 与 intersected_df 求交集

    # clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(init_num)+'_'+adapter)
    # clfdata = pd.read_csv(clf_file)
    # clfheader = generate_clf_header(label)
    #
    #
    #
    # # clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name =\
    # #     split_train_test(clfdata,clfheader,label,self_seed=9958)
    # clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name = \
    #     split_train_test(clfdata,clfheader,label)
    # clf_x_all = pd.concat([clf_x_test,clf_x_valid])
    # clf_y_all = pd.concat([clf_y_test,clf_y_valid])
    # clf_x_all.reset_index(inplace=True, drop=True)
    # clf_y_all.reset_index(inplace=True, drop=True)
    #
    # clf_all_ddf = pd.concat([clf_x_all,clf_y_all],axis= 1 )
    # clf_all_ddf =  clf_all_ddf[['dataname','opsname']]
    # intersected_df = pd.merge(intersected_df,clf_all_ddf,on=['dataname','opsname'] )
    # intersected_y = intersected_df['indexname'].copy(deep= True)
    #
    # intersect_num = len(intersected_y)

    feature_name  = list(x_train.columns)
    print('*'*90)
    print('the number of features is ', len(feature_name))
    print('lable: ', label)
    print('train: ',len(x_train), ' test: ',len(x_test),' valid: ',len(x_valid) )


    params_space = {'learning_rate': 0.3, 'n_estimators': 635, 'gamma': 0.0, 'max_depth': 8, 'min_child_weight': 2,
                    'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                    'seed': 9958,'objective':'reg:squarederror'}

    feature_num = len(feature_name)
    # all the sampling share the same parameters
    para_file = respath + 'params_hybrid_xgboost4regression_' + label + str(data_num) +sampling+ '.json'
    if adapter:
        para_file = respath + 'params_hybrid_xgboost4regression_' + label + str(28008) +sampling+ '.json'

    # each sampling has its own parameters
    # para_file = respath + 'params_hybrid_xgboost4regression_' + label + str(data_num) +'_' +sampling + '.json'
    if adapter:
        modelpath = respath+'model_hybrid_xgboost4regression_' + label + str(data_num) +'_'+str(feature_num)+'_'+sampling+'_'+adapter+'.pkl'
    else:
        modelpath = respath+'model_hybrid_xgboost4regression_' + label + str(data_num) +'_'+str(feature_num)+'_'+sampling+'.pkl'
    if (test_train==False and os.path.exists(modelpath)):
        reg_mod = joblib.load(modelpath)
        traintime = None
        if (os.path.exists(para_file)):
            with open(para_file,'r') as f:
                params_space = json.load(f)
    else:
        if (os.path.exists(para_file)):
            with open(para_file,'r') as f:
                params_space = json.load(f)
        else:
            # reg_mod = xgb.XGBRegressor(
            #     objective='reg:squarederror',
            #     max_depth=int(params_space['max_depth']),
            #     learning_rate=params_space['learning_rate'],
            #     subsample=params_space['subsample'],
            #     colsample_bytree=params_space['colsample_bytree'],
            #     n_estimators=int(params_space['n_estimators']),
            #     min_child_weight=int(params_space['min_child_weight']),
            #     tree_method='hist',
            #     gpu_id = '0'
            # )
            # reg_mod.fit(x_train, y_train, eval_metric='rmse', verbose=False)
            # selection = SelectFromModel(reg_mod, prefit=True,threshold=-np.inf,max_features=50)
            # print('特征选择阈值：', selection.threshold)
            # print('特征是否保留', selection.get_support())
            # choose_flag = selection.get_support()
            # idx = np.where(choose_flag==True)[0]
            # select_features = [feature_name[i] for i in idx]
            # params_space['select_feature'] = select_features
            # x_train = selection.transform(x_train)
            # x_test = selection.transform(x_test)
            # x_valid = selection.transform(x_valid)
            # print('selected features: ', select_features)
            # feature_num = len(params_space['select_feature'])

            params_space = {
                'n_estimators':hyperopt.hp.quniform("n_estimators",100,1500,15),
                'max_depth': hyperopt.hp.choice("max_depth",np.linspace(1, 10, 10, dtype=int)),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
                'reg_lambda':1,
                'subsample': 1,
                'reg_alpha': 0,
                'min_child_weight': hyperopt.hp.choice("min_child_weight",np.linspace(1, 10, 10, dtype=int)),
                'colsample_bytree': 1,
                'colsample_bylevel': 1,
                'gamma': 0,
                'tree_method': 'hist',
                'device': "cuda:0"

            }
            best = run_hypertuning(params_space,x_train,y_train,x_all,y_all)
            params_space.update(best)

            params_json = json.dumps(params_space,ensure_ascii=False, default=default_dump)
            with open(para_file,'w') as json_file:
                json_file.write(params_json)
        if label in []:  # 'bulkloadtime'
            reg_mod = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=int(params_space['max_depth']),
                learning_rate=params_space['learning_rate'],
                subsample=params_space['subsample'],
                colsample_bytree=params_space['colsample_bytree'],
                n_estimators=int(params_space['n_estimators']),
                min_child_weight=int(params_space['min_child_weight']),
                tree_method='hist',
                # gpu_id = '0'
            )
            reg_mod.fit(x_train, y_train, eval_metric='rmse', verbose=False)
            selection = SelectFromModel(reg_mod, prefit=True,threshold=-np.inf,max_features=100)
            print('特征选择阈值：', selection.threshold)
            print('特征是否保留', selection.get_support())
            choose_flag = selection.get_support()
            idx = np.where(choose_flag==True)[0]
            select_features = [feature_name[i] for i in idx]
            params_space['select_feature'] = select_features
            x_train = selection.transform(x_train)
            x_all = selection.transform(x_all)
            x_test = selection.transform(x_test)
            x_valid = selection.transform(x_valid)
            print('selected features: ', select_features)
            feature_num = len(params_space['select_feature'])
            reg_mod = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=int(params_space['max_depth']),
                learning_rate=params_space['learning_rate'],
                subsample=params_space['subsample'],
                colsample_bytree=params_space['colsample_bytree'],
                n_estimators=int(params_space['n_estimators']),
                min_child_weight=int(params_space['min_child_weight']),
                # gpu_id = '0'
            )
            start = time.perf_counter()
            reg_mod.fit(x_train,y_train)
            end = time.perf_counter()
            traintime = end-start
            joblib.dump(reg_mod,modelpath)


            start = time.perf_counter_ns()
            y_pred_all = reg_mod.predict(x_all)
            end = time.perf_counter_ns()
            all_reference_time = end-start
            res_dict = evaluate_model(y_all,y_pred_all)

            # 应用模型进行回归, 并预测best_index
            res_file = respath+'result_xgboost4regression'
            if adapter:
                res_file += '_adapter'

            clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,reg_mod,intersected_df,selection)
            avg_regret = clf_regret/(intersect_num-not_find)

            res_dict = res_dict_write(res_dict,label,val_data,modeltype,feature_type,sampling,feature_num,traintime,reg_mod,
                                      adapter,data_num,0,valdata_accuracy,all_reference_time,
                                      clf_regret,intersect_num,avg_regret,not_find)

            print (res_dict)
            write_res(res_file,res_dict,'header')

            return modelpath,para_file,data_num


        reg_mod = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=int(params_space['max_depth']),
            learning_rate=params_space['learning_rate'],
            subsample=params_space['subsample'],
            colsample_bytree=params_space['colsample_bytree'],
            n_estimators=int(params_space['n_estimators']),
            min_child_weight=int(params_space['min_child_weight']),
            # gpu_id = '0'
        )
        start = time.perf_counter()
        reg_mod.fit(x_train,y_train)
        end = time.perf_counter()
        traintime = end-start
        joblib.dump(reg_mod,modelpath)


    start = time.perf_counter_ns()
    y_pred_all = reg_mod.predict(x_all)
    end = time.perf_counter_ns()
    all_reference_time = end-start
    res_dict = evaluate_model(y_all,y_pred_all)

    # 应用模型进行回归, 并预测best_index
    res_file = respath+'result_xgboost4regression'
    if adapter:
        res_file += '_adapter'

    # clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,reg_mod,intersected_df,x_train.columns)
    # avg_regret = clf_regret/(intersect_num-not_find)
    #
    # res_dict = res_dict_write(res_dict,label,val_data,modeltype,feature_type,sampling,feature_num,traintime,reg_mod,
    #                           adapter,data_num,0,valdata_accuracy,all_reference_time,
    #                           clf_regret,intersect_num,avg_regret,not_find)
    #
    # print (res_dict)
    # write_res(res_file,res_dict,'header')

    return modelpath,para_file,data_num


# performance use the original value, without normalization
def True_value_Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,
                     feature_type = 'hybridencoder',modeltype = 'singleXGB',
                     adapter=None,test_train = True,val_data = 'finish'):


    if adapter:
        config_file = 'adapt_'+adapter+'_dataset_config.yaml'
    else:
        config_file = 'hasops_lias_dataset_config.yaml'

    args = {'config': './config/','configfile':'res_model_config.yaml',
            'dataset_id' : 'res_mtl',
            'datasetconfigfile':config_file}
    data_params = load_dataset_config(args['config'], args['dataset_id'],args['datasetconfigfile'])
    seed_everything(data_params['seed'])

    feature_encoder = FeatureProcessor(**data_params)

    # 加载 embedding
    # if label not in ['indexsize','bulkloadtime']:
    if label not in []:
        embedings = pd.read_csv(embeding_file)
        embed_head = []
        for i in range(32):
            embed_head.append('em'+str(i))
        embedings = embedings[embed_head]
    if 'stratified_' not in sampling:
        ratios = pd.read_csv(ratio_file)
        ratio_head = ['i','rq']
        ratios = ratios[ratio_head]
    # 把embeding 和 manual feature 合并
    if adapter:
        # if label not in ['indexsize','bulkloadtime']:
        if label not in []:
            if 'stratified_' in sampling:
                x_train,y_train,x_test,y_test,x_valid,y_valid = build_hybrid_init_dataset(feature_encoder, label,embedings,
                                                                                          **data_params)
            else:

                x_train,y_train,x_test,y_test,x_valid,y_valid = \
                    build_hybrid_init_sample_r_dataset(feature_encoder, label,embedings,
                                                       ratios,**data_params)
        else:
            if 'stratified_' in sampling:
                x_train,y_train,x_test,y_test,x_valid,y_valid = \
                    build_init_dataset(feature_encoder, label,**data_params)
            else:
                x_train,y_train,x_test,y_test,x_valid,y_valid = \
                    build_init_sample_r_dataset(feature_encoder, label,**data_params)
        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)
    else:
        # if label not in ["indexsize", "bulkloadtime"]:
        if label not in []:


            if feature_type == "naiveWorkloadFeature":
                if 'stratified_' in sampling:
                    x_train,y_train,x_test,y_test,x_valid,y_valid = build_dataset(feature_encoder, label,**data_params)
                else:
                    x_train,y_train,x_test,y_test,x_valid,y_valid = build_sample_r_dataset(feature_encoder, label,ratios,**data_params)

            else:
                if 'stratified_' in sampling:
                    x_train,y_train,x_test,y_test,x_valid,y_valid= build_hybrid_dataset(feature_encoder, label,embedings,
                                                                                        **data_params)
                else:
                    x_train,y_train,x_test,y_test,x_valid,y_valid= build_original_value_hybrid_sample_r_dataset(feature_encoder, label,embedings,ratios,
                                                                                                 **data_params)

        else:
            # x_train,y_train,x_test,y_test,x_valid,y_valid = build_dataset(feature_encoder, label,**data_params)
            x_train,y_train,x_test,y_test,x_valid,y_valid = build_sample_r_dataset(feature_encoder, label,ratios,**data_params)
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
        elif feature_type == "index_f_naive":
            drop_x_fearture = generate_index_header()
            drop_x_fearture = drop_x_fearture[1:]
            x_train = x_train.drop(drop_x_fearture, axis=1)
            x_test = x_test.drop(drop_x_fearture, axis=1)
            x_valid = x_valid.drop(drop_x_fearture, axis=1)
            print('Lord, chao needs You!')
        else:
            print('feature_type: ' , feature_type)

        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)

    data_num = len(x_train) + len(x_test) + len(x_valid)

    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']].copy(deep = True)
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)
    all_ddf = all_ddf.drop_duplicates(subset=['dataname','opsname'])
    # 验证 regret
    clfheader = generate_clf_header(label)
    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valdata = total_valdata[clfheader]
    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )

    # 加载一组分类数据， 划分train 和test, 与 intersected_df 求交集
    if adapter:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num)+'_init_'+adapter)
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)
    else:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num))
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)


    # clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name =\
    #     split_train_test(clfdata,clfheader,label,self_seed=9958)
    clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name = \
        split_train_test(clfdata,clfheader,label)
    clf_x_all = pd.concat([clf_x_test,clf_x_valid])
    clf_y_all = pd.concat([clf_y_test,clf_y_valid])
    clf_x_all.reset_index(inplace=True, drop=True)
    clf_y_all.reset_index(inplace=True, drop=True)

    clf_all_ddf = pd.concat([clf_x_all,clf_y_all],axis= 1 )
    clf_all_ddf =  clf_all_ddf[['dataname','opsname']]
    intersected_df = pd.merge(intersected_df,clf_all_ddf,on=['dataname','opsname'] )
    intersected_y = intersected_df['indexname'].copy(deep= True)

    intersect_num = len(intersected_y)

    feature_name  = list(x_train.columns)
    print('*'*90)
    print('the number of features is ', len(feature_name))
    print('lable: ', label)
    print('train: ',len(x_train), ' test: ',len(x_test),' valid: ',len(x_valid) )


    params_space = {'learning_rate': 0.3, 'n_estimators': 635, 'gamma': 0.0, 'max_depth': 8, 'min_child_weight': 2,
                    'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                    'seed': 9958,'objective':'reg:squarederror'}

    feature_num = len(feature_name)
    # all the sampling share the same parameters
    para_file = respath + 'params_hybrid_xgboost4regression_' + label + str(data_num) +sampling+ '.json'

    # each sampling has its own parameters
    # para_file = respath + 'params_hybrid_xgboost4regression_' + label + str(data_num) +'_' +sampling + '.json'
    if adapter:
        modelpath = respath+'model_original_hybrid_xgboost4regression_' + label + str(data_num) +'_'+str(feature_num)+'_'+sampling+'_'+adapter+'.pkl'
    else:
        modelpath = respath+'model_original_hybrid_xgboost4regression_' + label + str(data_num) +'_'+str(feature_num)+'_'+sampling+'.pkl'
    if (test_train==False and os.path.exists(modelpath)):
        reg_mod = joblib.load(modelpath)
        traintime = None
        if (os.path.exists(para_file)):
            with open(para_file,'r') as f:
                params_space = json.load(f)
    else:
        if (os.path.exists(para_file)):
            with open(para_file,'r') as f:
                params_space = json.load(f)
        else:
            # reg_mod = xgb.XGBRegressor(
            #     objective='reg:squarederror',
            #     max_depth=int(params_space['max_depth']),
            #     learning_rate=params_space['learning_rate'],
            #     subsample=params_space['subsample'],
            #     colsample_bytree=params_space['colsample_bytree'],
            #     n_estimators=int(params_space['n_estimators']),
            #     min_child_weight=int(params_space['min_child_weight']),
            #     tree_method='hist',
            #     gpu_id = '0'
            # )
            # reg_mod.fit(x_train, y_train, eval_metric='rmse', verbose=False)
            # selection = SelectFromModel(reg_mod, prefit=True,threshold=-np.inf,max_features=50)
            # print('特征选择阈值：', selection.threshold)
            # print('特征是否保留', selection.get_support())
            # choose_flag = selection.get_support()
            # idx = np.where(choose_flag==True)[0]
            # select_features = [feature_name[i] for i in idx]
            # params_space['select_feature'] = select_features
            # x_train = selection.transform(x_train)
            # x_test = selection.transform(x_test)
            # x_valid = selection.transform(x_valid)
            # print('selected features: ', select_features)
            # feature_num = len(params_space['select_feature'])

            params_space = {
                'n_estimators':hyperopt.hp.quniform("n_estimators",100,1500,15),
                'max_depth': hyperopt.hp.choice("max_depth",np.linspace(1, 10, 10, dtype=int)),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
                'reg_lambda':1,
                'subsample': 1,
                'reg_alpha': 0,
                'min_child_weight': hyperopt.hp.choice("min_child_weight",np.linspace(1, 10, 10, dtype=int)),
                'colsample_bytree': 1,
                'colsample_bylevel': 1,
                'gamma': 0,
                'tree_method': 'hist',
                'device': "cuda:0"

            }
            best = run_hypertuning(params_space,x_train,y_train,x_all,y_all)
            params_space.update(best)

            params_json = json.dumps(params_space,ensure_ascii=False, default=default_dump)
            with open(para_file,'w') as json_file:
                json_file.write(params_json)
        if label in []:  # 'bulkloadtime'
            reg_mod = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=int(params_space['max_depth']),
                learning_rate=params_space['learning_rate'],
                subsample=params_space['subsample'],
                colsample_bytree=params_space['colsample_bytree'],
                n_estimators=int(params_space['n_estimators']),
                min_child_weight=int(params_space['min_child_weight']),
                tree_method='hist',
                # gpu_id = '0'
            )
            reg_mod.fit(x_train, y_train, eval_metric='rmse', verbose=False)
            selection = SelectFromModel(reg_mod, prefit=True,threshold=-np.inf,max_features=100)
            print('特征选择阈值：', selection.threshold)
            print('特征是否保留', selection.get_support())
            choose_flag = selection.get_support()
            idx = np.where(choose_flag==True)[0]
            select_features = [feature_name[i] for i in idx]
            params_space['select_feature'] = select_features
            x_train = selection.transform(x_train)
            x_all = selection.transform(x_all)
            x_test = selection.transform(x_test)
            x_valid = selection.transform(x_valid)
            print('selected features: ', select_features)
            feature_num = len(params_space['select_feature'])
            reg_mod = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=int(params_space['max_depth']),
                learning_rate=params_space['learning_rate'],
                subsample=params_space['subsample'],
                colsample_bytree=params_space['colsample_bytree'],
                n_estimators=int(params_space['n_estimators']),
                min_child_weight=int(params_space['min_child_weight']),
                # gpu_id = '0'
            )
            start = time.perf_counter()
            reg_mod.fit(x_train,y_train)
            end = time.perf_counter()
            traintime = end-start
            joblib.dump(reg_mod,modelpath)


            start = time.perf_counter_ns()
            y_pred_all = reg_mod.predict(x_all)
            end = time.perf_counter_ns()
            all_reference_time = end-start
            res_dict = evaluate_model(y_all,y_pred_all)

            # 应用模型进行回归, 并预测best_index
            res_file = respath+'result_xgboost4regression'
            if adapter:
                res_file += '_adapter'

            clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,reg_mod,intersected_df,selection)
            avg_regret = clf_regret/(intersect_num-not_find)

            res_dict = res_dict_write(res_dict,label,val_data,modeltype,feature_type,sampling,feature_num,traintime,reg_mod,
                                      adapter,data_num,0,valdata_accuracy,all_reference_time,
                                      clf_regret,intersect_num,avg_regret,not_find)

            print (res_dict)
            write_res(res_file,res_dict,'header')

            return modelpath,para_file,data_num


        reg_mod = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=int(params_space['max_depth']),
            learning_rate=params_space['learning_rate'],
            subsample=params_space['subsample'],
            colsample_bytree=params_space['colsample_bytree'],
            n_estimators=int(params_space['n_estimators']),
            min_child_weight=int(params_space['min_child_weight']),
            # gpu_id = '0'
        )
        start = time.perf_counter()
        reg_mod.fit(x_train,y_train)
        end = time.perf_counter()
        traintime = end-start
        joblib.dump(reg_mod,modelpath)


    start = time.perf_counter_ns()
    y_pred_all = reg_mod.predict(x_all)
    end = time.perf_counter_ns()
    all_reference_time = end-start
    res_dict = evaluate_model(y_all,y_pred_all)

    # 应用模型进行回归, 并预测best_index
    res_file = respath+'result_original_xgboost4regression'
    if adapter:
        res_file += '_adapter'

    clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,reg_mod,intersected_df,x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,val_data,modeltype,feature_type,sampling,feature_num,traintime,reg_mod,
                              adapter,data_num,0,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)

    print (res_dict)
    write_res(res_file,res_dict,'header')

    return modelpath,para_file,data_num


# 应用hyperopt 进行调参
def Hybrid_RFRegressor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,
                       feature_type = 'hybridencoder',modeltype = 'singleRF',adapter=None,
                       test_train = True,val_data = 'finish'):


    if adapter:
        config_file = 'adapt_'+adapter+'_dataset_config.yaml'
    else:
        config_file = 'hasops_lias_dataset_config.yaml'

    args = {'config': './config/','configfile':'res_model_config.yaml',
            'dataset_id' : 'res_mtl',
            'datasetconfigfile':config_file}
    data_params = load_dataset_config(args['config'], args['dataset_id'],args['datasetconfigfile'])
    seed_everything(data_params['seed'])

    feature_encoder = FeatureProcessor(**data_params)


    if label not in []:
        embedings = pd.read_csv(embeding_file)
        embed_head = []
        for i in range(32):
            embed_head.append('em'+str(i))
        embedings = embedings[embed_head]
    if 'stratified_' not in sampling:
        ratios = pd.read_csv(ratio_file)
        ratio_head = ['i','rq']
        ratios = ratios[ratio_head]
        # 把embeding 和 manual feature 合并
    if adapter:
        # if label not in ['indexsize','bulkloadtime']:
        if label not in []:
            x_train,y_train,x_test,y_test,x_valid,y_valid = build_hybrid_init_dataset(feature_encoder, label,embedings,
                                                                                      **data_params)
        else:
            x_train,y_train,x_test,y_test,x_valid,y_valid = build_init_dataset(feature_encoder, label,**data_params)
        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)
    else:
        # if label not in ["indexsize", "bulkloadtime"]:
        if label not in []:

            if feature_type == "naiveWorkloadFeature":
                if 'stratified_' in sampling:
                    x_train,y_train,x_test,y_test,x_valid,y_valid = build_dataset(feature_encoder, label,**data_params)
                else:
                    x_train,y_train,x_test,y_test,x_valid,y_valid = build_sample_r_dataset(feature_encoder, label,ratios,**data_params)

            else:
                if 'stratified_' in sampling:
                    x_train,y_train,x_test,y_test,x_valid,y_valid= build_hybrid_dataset(feature_encoder, label,embedings,
                                                                                        **data_params)
                else:
                    x_train,y_train,x_test,y_test,x_valid,y_valid= build_hybrid_sample_r_dataset(feature_encoder, label,embedings,ratios,
                                                                                                 **data_params)

        else:
            # x_train,y_train,x_test,y_test,x_valid,y_valid = build_dataset(feature_encoder, label,**data_params)
            x_train,y_train,x_test,y_test,x_valid,y_valid = build_sample_r_dataset(feature_encoder, label,ratios,**data_params)

        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)


    data_num = len(x_train) + len(x_test) + len(x_valid)
    feature_name  = list(x_train.columns)
    feature_num = len(feature_name)
    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']].copy(deep = True)
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)
    all_ddf = all_ddf.drop_duplicates(subset=['dataname','opsname'])
    # 验证 regret
    clfheader = generate_clf_header(label)
    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valdata = total_valdata[clfheader]
    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )

    # 加载一组分类数据， 划分train 和test, 与 intersected_df 求交集
    if adapter:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num)+'_'+adapter)
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)
    else:
        clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(reg_data_num))
        clfdata = pd.read_csv(clf_file)
        clfheader = generate_clf_header(label)


    # clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name =\
    #     split_train_test(clfdata,clfheader,label,self_seed=9958)
    clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name = \
        split_train_test(clfdata,clfheader,label)
    clf_x_all = pd.concat([clf_x_test,clf_x_valid])
    clf_y_all = pd.concat([clf_y_test,clf_y_valid])
    clf_x_all.reset_index(inplace=True, drop=True)
    clf_y_all.reset_index(inplace=True, drop=True)

    clf_all_ddf = pd.concat([clf_x_all,clf_y_all],axis= 1 )
    clf_all_ddf =  clf_all_ddf[['dataname','opsname']]
    intersected_df = pd.merge(intersected_df,clf_all_ddf,on=['dataname','opsname'] )
    intersected_y = intersected_df['indexname'].copy(deep= True)

    intersect_num = len(intersected_y)

    feature_name  = list(x_train.columns)
    print('*'*90)
    print('the number of features is ', len(feature_name))
    print('lable: ', label)
    print('train: ',len(x_train), ' test: ',len(x_test),' valid: ',len(x_valid) )


    feature_name  = list(x_train.columns)
    print('*'*90)
    print('feature_name: ', feature_name)
    print('the number of features is ', len(feature_name))
    print('lable: ', label)
    print('train: ',len(x_train),len(y_train))
    print('test: ',len(x_test),len(y_test))
    print('valid: ',len(x_valid),len(y_valid))

    # other_params = {'eta': 0.3, 'n_estimators': 1500, 'gamma': 0.0, 'max_depth': 6, 'min_child_weight': 1,
    #                 'booster': 'gbtree', 'colsample_bynode': 0.8, 'subsample': 0.8, "tree_method": "hist",
    #                 'seed': 9958}


    select_feature_num = len(feature_name)
    if adapter:
        modelpath = respath+'model_hybrid_RF4regression_' + label + str(data_num) +'_'+str(select_feature_num)+'_'+adapter+'.pkl'
    else:
        modelpath = respath+'model_hybrid_RF4regression_' + label + str(data_num) +'_'+str(select_feature_num)+'.pkl'
    if (test_train==False and os.path.exists(modelpath)):
        para_file = respath+'params_hybrid_RF4regression_'+ label +str(data_num)+'.json'
        reg_mod = joblib.load(modelpath)
        traintime = None
    else:
        # para_file = respath+'params_hybrid_xgboost4regression_'+ label +str(data_num)+ \
        #             '_'+str(select_feature_num)+'.json'
        para_file = respath+'params_hybrid_RF4regression_'+ label +str(data_num)+'.json'
        if (os.path.exists(para_file)):
            with open(para_file,'r') as f:
                params_space = json.load(f)
        else:

            params_space = {
                'n_estimators':hyperopt.hp.quniform("n_estimators",100,1500,15),
                'max_depth': hyperopt.hp.choice("max_depth",np.linspace(11, 20, 10, dtype=int)),
                'min_samples_split':hyperopt.hp.uniform('min_samples_split',2,6),
                'min_samples_leaf':hyperopt.hp.uniform('min_samples_leaf',1,5),
                'oob_score':True,
                'n_jobs': -1,
                'seed':9958
            }
            best = run_RF_hypertuning(params_space,x_train,y_train,x_all,y_all)
            params_space.update(best)

            params_json = json.dumps(params_space,ensure_ascii=False, default=default_dump)
            with open(para_file,'w') as json_file:
                json_file.write(params_json)

        reg_mod = RandomForestRegressor(
            max_depth=int(params_space['max_depth']),
            min_samples_leaf=int(params_space['min_samples_leaf']),
            min_samples_split=int(params_space['min_samples_split']),
            n_estimators=int(params_space['n_estimators']),
            n_jobs= params_space['n_jobs'],
            warm_start=True
        )


        start = time.perf_counter()
        reg_mod.fit(x_train,y_train)
        end = time.perf_counter()
        traintime = end-start
        joblib.dump(reg_mod,modelpath)
    # 应用模型进行回归


    start = time.perf_counter_ns()
    y_pred_all = np.abs(reg_mod.predict(x_all))
    end = time.perf_counter_ns()
    all_reference_time = end-start
    res_file = respath+'result_xgboost4regression'
    res_dict = evaluate_model(y_all,y_pred_all)
    clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,reg_mod,intersected_df,x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,val_data,modeltype,feature_type,sampling,feature_num,traintime,reg_mod,
                              adapter,data_num,0,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)

    print (res_dict)
    write_res(res_file,res_dict,'header')

    return modelpath,para_file,data_num


# 用于增量更新已经训练好的random forest regressor
def incremental_single_RF(label,modelpath,parapath,adapter,featuretype,sampling, modeltype= 'singleRF',hybrid = False,adpat_embeding_file= None,init_embeding_file=None):

    if (os.path.exists(parapath)):
        with open(parapath,'r') as f:
            other_params = json.load(f)
    else:
        print ("Lord, please come! the params does not exist!")
        return

    config_file = 'adapt_'+adapter+'_dataset_config.yaml'
    args = {'config': './config/','configfile':'res_model_config.yaml',
            'dataset_id' : 'res_mtl',
            'datasetconfigfile':config_file}
    data_params = load_dataset_config(args['config'], args['dataset_id'],args['datasetconfigfile'])
    seed_everything(data_params['seed'])
    # 加载 new collected data
    feature_encoder = FeatureProcessor(**data_params)
    if hybrid:
        adpat_embeding_file += ('_'+adapter)
        adaptembedings = pd.read_csv(adpat_embeding_file)
        embed_head = []
        for i in range(32):
            embed_head.append('em'+str(i))
        adaptembedings = adaptembedings[embed_head]
        x_train,y_train,x_test,y_test,x_valid,y_valid = build_hybrid_adapt_dataset(feature_encoder, label,adaptembedings,**data_params)
        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)
        # 加载embedings
        init_embeding_file += ('_'+adapter)
        initembedings = pd.read_csv(init_embeding_file)
        embed_head = []
        for i in range(32):
            embed_head.append('em'+str(i))
        initembedings = initembedings[embed_head]
        init_x_train,init_y_train,init_x_test,init_y_test,init_x_valid,init_y_valid  = build_hybrid_init_dataset(feature_encoder, label,initembedings,
                                                                                                                 **data_params)
        init_x_all = pd.concat([init_x_test,init_x_valid])
        init_y_all = pd.concat([init_y_test,init_y_valid])
        init_x_all.reset_index(inplace=True, drop=True)
        init_y_all.reset_index(inplace=True, drop=True)

    else:

        x_train,y_train,x_test,y_test,x_valid,y_valid = build_adapt_dataset(feature_encoder, label,**data_params)
        x_all = pd.concat([x_test,x_valid])
        y_all = pd.concat([y_test,y_valid])
        x_all.reset_index(inplace=True, drop=True)
        y_all.reset_index(inplace=True, drop=True)
        # 加载 init data
        init_x_train,init_y_train,init_x_test,init_y_test,init_x_valid,init_y_valid = build_init_dataset(feature_encoder, label,**data_params)
        init_x_all = pd.concat([init_x_test,init_x_valid])
        init_y_all = pd.concat([init_y_test,init_y_valid])
        init_x_all.reset_index(inplace=True, drop=True)
        init_y_all.reset_index(inplace=True, drop=True)
    init_x_num = len(init_x_train) + len(init_x_test) + len(init_x_valid)
    incremental_x_num = len(x_train) + len(x_test) + len(x_valid)

    feature_name  = list(x_train.columns)
    print('*'*90)
    print('feature_name: ', feature_name)
    print('lable: ', label)
    print('train: ',len(x_train),len(y_train))
    print('test: ',len(x_test),len(y_test))
    feature_num = len(feature_name)
    res_file = respath+'result_xgboost4regression_adapter'


    # 加载已经训练好的模型
    init_model = load_model(modelpath,label)

    # 测试 init model 在init data 上的性能

    y_valid_pred = np.abs(init_model.predict(init_x_valid))
    res_dict = evaluate_model(init_y_valid,y_valid_pred)
    res_dict['task'] = label
    res_dict['data'] = 'init_valid_data'
    res_dict['model'] = 'init_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = None
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    print ('the validation performance of RFregressor are: ')
    print (res_dict)

    write_res(res_file, res_dict, 'header')

    y_test_pred = np.abs(init_model.predict(init_x_test))
    res_dict = evaluate_model(init_y_test,y_test_pred)
    res_dict['task'] = label
    res_dict['data'] = 'init_test_data'
    res_dict['model'] = 'init_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = None
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    print ('the test performance of xgregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    y_all_pred = np.abs(init_model.predict(init_x_all))
    res_dict = evaluate_model(init_y_all,y_all_pred)
    res_dict['task'] = label
    res_dict['data'] = 'init_all_data'
    res_dict['model'] = 'init_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = None
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    print ('the test performance of xgregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)


    # 1. 测试一下 newdata 在init model 上的性能
    # validation

    y_valid_pred = np.abs(init_model.predict(x_valid))
    res_dict = evaluate_model(y_valid,y_valid_pred)
    res_dict['task'] = label
    res_dict['data'] = 'new_valid_data'
    res_dict['model'] = 'init_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = None
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    print ('the validation performance of xgregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)


    y_pred = np.abs(init_model.predict(x_test))
    res_dict = evaluate_model(y_test,y_pred)
    res_dict['task'] = label
    res_dict['data'] = 'new_test_data'
    res_dict['model'] = 'init_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = None
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    # 输出模型性能，并保存
    print ('the test performance of RFregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    y_all_pred = np.abs(init_model.predict(x_all))
    res_dict = evaluate_model(y_all,y_all_pred)
    res_dict['task'] = label
    res_dict['data'] = 'new_all_data'
    res_dict['model'] = 'init_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = None
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    # 输出模型性能，并保存
    print ('the test performance of RFregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    # incremental learning new data
    print('init_model.estimators: ',init_model.n_estimators)
    init_model.n_estimators += 100
    start = time.perf_counter()
    init_model.fit(x_train,y_train)
    end = time.perf_counter()
    print('adpat model.estimators: ',init_model.n_estimators)
    print ('Lord, chao needs You, from ever to ever!')

    adapttime = end-start


    # xgb_model (str | PathLike | Booster | bytearray | None) –
    # Xgb model to be loaded before training (allows training continuation).
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.fit
    # Note that calling fit() multiple times will cause the model object to be re-fit from scratch.
    # To resume training from a previous checkpoint, explicitly pass xgb_model argument.
    print('Lord, thank You for your company!!')

    # 2. 测试incremental model 的效果 在new data 上的效果
    y_valid_pred = np.abs(init_model.predict(x_valid))
    y_test_pred = np.abs(init_model.predict(x_test))
    y_all_pred = np.abs(init_model.predict(x_all))

    # 评价性能
    res_dict = evaluate_model(y_test,y_test_pred)
    res_dict['task'] = label
    res_dict['data'] = 'new_test_data'
    res_dict['model'] = 'adapt_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = adapttime

    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    # 输出模型性能，并保存
    print ('the test performance of RFregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    res_dict = evaluate_model(y_valid,y_valid_pred)
    res_dict['task'] = label
    res_dict['data'] = 'new_valid_data'
    res_dict['model'] = 'adapt_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = adapttime
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    # 输出模型性能，并保存
    print ('the test performance of RFregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    res_dict = evaluate_model(y_all,y_all_pred)
    res_dict['task'] = label
    res_dict['data'] = 'new_all_data'
    res_dict['model'] = 'adapt_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = adapttime
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    # 输出模型性能，并保存
    print ('the test performance of xgregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    # 测试 new model 在init data 上的性能
    y_valid_pred = np.abs(init_model.predict(init_x_valid))
    y_test_pred = np.abs(init_model.predict(init_x_test))
    y_all_pred = np.abs(init_model.predict(init_x_all))

    res_dict = evaluate_model(init_y_valid,y_valid_pred)
    res_dict['task'] = label

    res_dict['data'] = 'init_valid_data'
    res_dict['model'] = 'adapt_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = adapttime
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    print ('the validation performance of regressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    print('Lord, thank You! You are my refuge from ever to ever')

    res_dict = evaluate_model(init_y_test,y_test_pred)
    res_dict['task'] = label

    res_dict['data'] = 'init_test_data'
    res_dict['model'] = 'adapt_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = adapttime
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    print ('the test performance of xgregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    res_dict = evaluate_model(init_y_all,y_all_pred)
    res_dict['task'] = label

    res_dict['data'] = 'init_all_data'
    res_dict['model'] = 'adapt_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = adapttime
    res_dict['n_estimators'] = init_model.n_estimators
    res_dict['min_samples_split'] = init_model.min_samples_split
    res_dict['max_depth'] = init_model.max_depth
    res_dict['min_samples_leaf'] = init_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    print ('the validation performance of regressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    # retrain a new model from scratch
    # merge init_data and new data, retrain from scratch
    merge_x_train = pd.concat([init_x_train,x_train,init_x_test,init_x_valid])
    merge_y_train = pd.concat([init_y_train,y_train,init_y_test,init_y_valid])
    merge_train = pd.concat([merge_x_train,merge_y_train],axis=1).reset_index()
    all_x_train,all_y_train,all_x_test,all_y_test,all_x_valid,all_y_valid = build_retrain_dataset(merge_train,label)


    # build a new model
    new_model = RandomForestRegressor(
        max_depth=init_model.max_depth,
        min_samples_leaf=init_model.min_samples_leaf,
        min_samples_split=init_model.min_samples_split,
        n_estimators=init_model.n_estimators,
        n_jobs= init_model.n_jobs,
        warm_start=True
    )


    start = time.perf_counter()
    new_model.fit(all_x_train,all_y_train)
    end = time.perf_counter()
    retraintime = end-start
    newmodelpath = list(modelpath)
    newmodelpath.insert(-4,'_retrain')
    newmodelpath = ''.join(newmodelpath)
    joblib.dump(new_model,newmodelpath)
    # 应用模型进行回归



    y_all_pred = np.abs(new_model.predict(x_all))

    res_dict = evaluate_model(y_all,y_all_pred)
    res_dict['task'] = label
    res_dict['data'] = 'new_all_data'
    res_dict['model'] = 'new_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = retraintime
    res_dict['n_estimators'] = new_model.n_estimators
    res_dict['min_samples_split'] = new_model.min_samples_split
    res_dict['max_depth'] = new_model.max_depth
    res_dict['min_samples_leaf'] = new_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    # 输出模型性能，并保存
    print ('the test performance of RFregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)

    y_init_all_pred = np.abs(new_model.predict(init_x_all))
    res_dict = evaluate_model(init_y_all,y_init_all_pred)
    res_dict['task'] = label
    res_dict['data'] = 'init_all_data'
    res_dict['model'] = 'new_'+modeltype
    res_dict['feature'] = featuretype
    res_dict['sampling'] = sampling
    res_dict['featurenum'] = feature_num
    res_dict['traintime'] = retraintime
    res_dict['n_estimators'] = new_model.n_estimators
    res_dict['min_samples_split'] = new_model.min_samples_split
    res_dict['max_depth'] = new_model.max_depth
    res_dict['min_samples_leaf'] = new_model.min_samples_leaf
    res_dict['adapter'] = adapter
    res_dict['init_x_num'] = init_x_num
    res_dict['incremental_x_num'] = incremental_x_num
    # 输出模型性能，并保存
    print ('the test performance of RFregressor are: ')
    print (res_dict)
    write_res(res_file, res_dict)


def load_model(modelpath,label='indexsize'):

    assert (os.path.exists(modelpath)), label+" model does not exist"
    reg_mod = joblib.load(modelpath)

    return reg_mod


# 用于增量更新已经训练好的XGB regressor
# (label,modelpath,parapath,adapter,feature_type,sampling,allsampling,v_data_num,
#  init_embeding_file,adapt_embeding_file,init_ratio_file,adapt_ratio_file
def incremental_single_xgb(label,modelpath,parapath,adapter,featuretype,sampling,allsampling,init_num,adapt_num,
                           v_data_num,init_embeding_file,adapt_embeding_file,init_ratio_file,adapt_ratio_file,
                           modeltype= 'singleXGB',val_data = 'finish'):
    # 加载已经训练好的模型

    init_model = load_model(modelpath,label)
    if (os.path.exists(parapath)):
        with open(parapath,'r') as f:
            other_params = json.load(f)
    else:
        print ("Lord, please come! the params does not exist!")
        return

    seed_everything(9958)
    path = '/home/wamdm/chaohong/index_selection/total_init_adapt/'
    adaptdatafile = path+'normal_systematic_10thousand_total_feature_'+str(adapt_num)+'_'+adapter

    x_train,y_train,x_test,y_test,x_valid,y_valid = new_init_dataset(label,adaptdatafile)

    x_all = pd.concat([x_test,x_valid])
    y_all = pd.concat([y_test,y_valid])
    x_all.reset_index(inplace=True, drop=True)
    y_all.reset_index(inplace=True, drop=True)
    # 加载initdata
    initdatafile = path+'normal_systematic_10thousand_total_feature_'+str(init_num)+'_'+adapter
    init_x_train,init_y_train,init_x_test,init_y_test,init_x_valid,init_y_valid = new_init_dataset(label,initdatafile)

    init_x_all = pd.concat([init_x_test,init_x_valid])
    init_y_all = pd.concat([init_y_test,init_y_valid])
    init_x_all.reset_index(inplace=True, drop=True)
    init_y_all.reset_index(inplace=True, drop=True)



    init_x_num = len(init_x_train) + len(init_x_test) + len(init_x_valid)
    incremental_x_num = len(x_train) + len(x_test) + len(x_valid)


    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    # all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']].copy(deep = True)
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)
    all_ddf = all_ddf.drop_duplicates(subset=['dataname','opsname'])
    # 验证 regret
    clfheader = generate_clf_header(label)
    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valdata = total_valdata[clfheader]
    valdata = valdata[['dataname','opsname']]
    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )

    # 加载一组分类数据， 划分train 和test, 与 intersected_df 求交集
    clf_file = os.path.join(clf_datapath, label+'_'+allsampling+str(incremental_x_num)+'_'+adapter)
    clfdata = pd.read_csv(clf_file)
    clfheader = generate_clf_header(label)

    # clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name =\
    #     split_train_test(clfdata,clfheader,label,self_seed=9958)
    clf_x_train,clf_y_train,clf_x_test,clf_y_test,clf_x_valid,clf_y_valid,clf_num_class,clf_class_name = \
        split_train_test(clfdata,clfheader,label)
    clf_x_all = pd.concat([clf_x_test,clf_x_valid])
    clf_y_all = pd.concat([clf_y_test,clf_y_valid])
    clf_x_all.reset_index(inplace=True, drop=True)
    clf_y_all.reset_index(inplace=True, drop=True)

    clf_all_ddf = pd.concat([clf_x_all,clf_y_all],axis= 1 )
    clf_all_ddf =  clf_all_ddf[['dataname','opsname']]
    intersected_df = pd.merge(intersected_df,clf_all_ddf,on=['dataname','opsname'] )
    intersected_y = intersected_df['indexname'].copy(deep= True)

    intersect_num = len(intersected_y)

    feature_name  = list(x_train.columns)
    print('*'*90)
    print('feature_name: ', feature_name)
    print('lable: ', label)
    print('train: ',len(x_train),len(y_train))
    print('test: ',len(x_test),len(y_test))
    feature_num = len(feature_name)
    res_file = respath+'result_xgboost4regression_adapter'
    # 测试 init model 在new data 上的性能
    start = time.perf_counter_ns()
    y_pred_all = init_model.predict(x_all)
    end = time.perf_counter_ns()
    all_reference_time = end-start
    res_dict = evaluate_model(y_all,y_pred_all)
    clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,init_model,intersected_df,x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,'new_all_data','init_'+modeltype,featuretype,sampling,feature_num,None,init_model,
                              adapter,init_x_num,incremental_x_num,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)

    print (res_dict)
    write_res(res_file,res_dict,'header')


    # incremental learning data

    adapt_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=int(other_params['max_depth']),
        learning_rate=other_params['learning_rate'],
        subsample=other_params['subsample'],
        colsample_bytree=other_params['colsample_bytree'],
        n_estimators=int(other_params['n_estimators']),
        min_child_weight=int(other_params['min_child_weight']),
    )


    start = time.perf_counter()
    adapt_model.fit(x_train,y_train,xgb_model=init_model.get_booster())
    end = time.perf_counter()
    adapttime = end-start


    # xgb_model (str | PathLike | Booster | bytearray | None) –
    # Xgb model to be loaded before training (allows training continuation).
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.fit
    # Note that calling fit() multiple times will cause the model object to be re-fit from scratch.
    # To resume training from a previous checkpoint, explicitly pass xgb_model argument.
    print('Lord, thank You for your company!!')

    # 2. 测试incremental model 的效果 在new data 上的效果

    y_all_pred = np.abs(adapt_model.predict(x_all))
    res_dict = evaluate_model(y_all,y_all_pred)
    clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,adapt_model,intersected_df,x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,'new_all_data','adapt_'+modeltype,featuretype,sampling,feature_num,adapttime,adapt_model,
                              adapter,init_x_num,incremental_x_num,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)

    print (res_dict)
    write_res(res_file,res_dict)

    # retrain a new model from scratch
    # merge init_data and new data, retrain from scratch
    init_x_train = init_x_train.drop(['dataname','opsname'], axis=1)
    init_x_test = init_x_test.drop(['dataname','opsname'], axis=1)
    init_x_valid = init_x_valid.drop(['dataname','opsname'], axis=1)
    merge_x_train = pd.concat([init_x_train,x_train,init_x_test,init_x_valid])
    merge_y_train = pd.concat([init_y_train,y_train,init_y_test,init_y_valid])
    merge_train = pd.concat([merge_x_train,merge_y_train],axis=1).reset_index()
    all_x_train,all_y_train,all_x_test,all_y_test,all_x_valid,all_y_valid = build_retrain_dataset(merge_train,label)

    # build a new model
    new_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=int(other_params['max_depth']),
        learning_rate=other_params['learning_rate'],
        subsample=other_params['subsample'],
        colsample_bytree=other_params['colsample_bytree'],
        n_estimators=int(other_params['n_estimators']),
        min_child_weight=int(other_params['min_child_weight']),
    )


    start = time.perf_counter()
    new_model.fit(all_x_train,all_y_train)
    end = time.perf_counter()
    retraintime = end-start
    newmodelpath = list(modelpath)
    newmodelpath.insert(-4,'_retrain')
    newmodelpath = ''.join(newmodelpath)
    joblib.dump(new_model,newmodelpath)
    # 应用模型进行回归
    y_all_pred = np.abs(new_model.predict(x_all))
    res_dict = evaluate_model(y_all,y_all_pred)
    clf_regret,not_find,valdata_accuracy = regret_single_objeectie_Selector(label,new_model,intersected_df,all_x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,'new_all_data','new_'+modeltype,featuretype,sampling,feature_num,retraintime,new_model,
                              adapter,init_x_num,incremental_x_num,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)

    print (res_dict)
    write_res(res_file,res_dict)

def index_incremental_single_xgb(label,modelpath,parapath,adapter,featuretype,sampling,allsampling,init_num,adapt_num,
                           v_data_num,init_embeding_file,adapt_embeding_file,init_ratio_file,adapt_ratio_file,
                           modeltype= 'singleXGB',val_data = 'finish'):
    # 加载已经训练好的模型
    seed_everything(9958)
    init_model = load_model(modelpath,label)
    if (os.path.exists(parapath)):
        with open(parapath,'r') as f:
            other_params = json.load(f)
    else:
        print ("Lord, please come! the params does not exist!")
        return

    seed_everything(9958)
    path = '/home/wamdm/chaohong/index_selection/total_init_adapt/'
    adaptdatafile = path+'normal_systematic_10thousand_total_feature_'+str(adapt_num)+'_'+adapter

    x_train,y_train,x_test,y_test,x_valid,y_valid = new_init_dataset(label,adaptdatafile)

    x_all = pd.concat([x_test,x_valid])
    y_all = pd.concat([y_test,y_valid])
    # y_all_change = y_all.apply(lambda x: x*10)
    x_all.reset_index(inplace=True, drop=True)
    y_all.reset_index(inplace=True, drop=True)
    # 加载initdata
    initdatafile = path+'normal_systematic_10thousand_total_feature_'+str(init_num)+'_'+adapter
    init_x_train,init_y_train,init_x_test,init_y_test,init_x_valid,init_y_valid = new_init_dataset(label,initdatafile)

    init_x_all = pd.concat([init_x_test,init_x_valid])
    init_y_all = pd.concat([init_y_test,init_y_valid])
    init_x_all.reset_index(inplace=True, drop=True)
    init_y_all.reset_index(inplace=True, drop=True)



    init_x_num = len(init_x_train) + len(init_x_test) + len(init_x_valid)
    incremental_x_num = len(x_train) + len(x_test) + len(x_valid)


    all_ddf = pd.concat([x_all,y_all],axis= 1 )
    all_ddf = all_ddf.drop_duplicates(subset=['dataname','opsname'])
    # all_ddf =  all_ddf[['dataname','opsname','is_rq','is_i','lookup']].copy(deep = True)
    x_all = x_all.drop(['dataname','opsname'], axis=1)
    x_train = x_train.drop(['dataname','opsname'], axis=1)
    x_test = x_test.drop(['dataname','opsname'], axis=1)
    x_valid = x_valid.drop(['dataname','opsname'], axis=1)
    # 验证 regret
    clfheader = generate_clf_header(label)
    val_file = os.path.join(val_datapath,label + '_' + val_data +'_file'+str(v_data_num))
    total_valdata = pd.read_csv(val_file)
    valdata = total_valdata[clfheader]
    valdata = valdata[['dataname','opsname']]
    intersected_df = pd.merge(valdata,all_ddf,on=['dataname','opsname'] )
    index_header = generate_index_header()
    intersected_df = intersected_df.drop(index_header,axis=1)
    intersected_y = intersected_df[label].copy(deep= True)

    intersect_num = len(intersected_y)

    feature_name  = list(x_train.columns)
    print('*'*90)
    print('feature_name: ', feature_name)
    print('lable: ', label)
    print('train: ',len(x_train),len(y_train))
    print('test: ',len(x_test),len(y_test))
    feature_num = len(feature_name)
    res_file = respath+'result_xgboost4regression_adapter'
    # 测试 init model 在new data 上的性能
    start = time.perf_counter_ns()
    y_pred_all = init_model.predict(x_all)

    end = time.perf_counter_ns()
    all_reference_time = end-start
    res_dict = evaluate_model(y_all,y_pred_all)
    clf_regret,not_find,valdata_accuracy,intersected_df = index_regret_single_objeectie_Selector(label,init_model,intersected_df,x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,'new_all_data','init_'+modeltype,featuretype,sampling,feature_num,None,init_model,
                              adapter,init_x_num,incremental_x_num,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)
    intersect_num = len(intersected_df)
    print (res_dict)
    write_res(res_file,res_dict,'header')

    # train local model

    local_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=int(other_params['max_depth']),
        learning_rate=other_params['learning_rate'],
        subsample=other_params['subsample'],
        colsample_bytree=other_params['colsample_bytree'],
        n_estimators=int(other_params['n_estimators']),
        min_child_weight=int(other_params['min_child_weight']),
    )


    start = time.perf_counter()
    num_round = 100
    local_model = local_model.fit(x_train,y_train,xgb_model=init_model.get_booster())
    end = time.perf_counter()
    localtraintime = end-start


    # merge init_data and new data, retrain from scratch
    init_x_train = init_x_train.drop(['dataname','opsname'], axis=1)
    init_x_test = init_x_test.drop(['dataname','opsname'], axis=1)
    init_x_valid = init_x_valid.drop(['dataname','opsname'], axis=1)
    merge_x_train = pd.concat([init_x_train,x_train,init_x_test,init_x_valid])

    merge_y_train = pd.concat([init_y_train,y_train,init_y_test,init_y_valid])
    merge_train = pd.concat([merge_x_train,merge_y_train],axis=1).reset_index()
    all_x_train,all_y_train,all_x_test,all_y_test,all_x_valid,all_y_valid = build_retrain_dataset(merge_train,label)


    # incremental learning data

    adapt_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=int(other_params['max_depth']),
        learning_rate=other_params['learning_rate'],
        subsample=other_params['subsample'],
        colsample_bytree=other_params['colsample_bytree'],
        n_estimators=int(other_params['n_estimators']),
        min_child_weight=int(other_params['min_child_weight']),
    )


    start = time.perf_counter()
    num_round = 100
    adapt_model = adapt_model.fit(all_x_train,all_y_train,xgb_model=init_model.get_booster())
    end = time.perf_counter()
    adapttime = end-start


    # xgb_model (str | PathLike | Booster | bytearray | None) –
    # Xgb model to be loaded before training (allows training continuation).
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.fit
    # Note that calling fit() multiple times will cause the model object to be re-fit from scratch.
    # To resume training from a previous checkpoint, explicitly pass xgb_model argument.
    print('Lord, thank You for your company!!')

    # 2. 测试incremental model 的效果 在new data 上的效果

    y_all_pred = adapt_model.predict(x_all)
    # y_all_pred = y_all_pred/100
    res_dict = evaluate_model(y_all,y_all_pred)

    clf_regret,not_find,valdata_accuracy,intersected_df = index_regret_single_objeectie_Selector(label,adapt_model,intersected_df,x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,'new_all_data','adapt_'+modeltype,featuretype,sampling,feature_num,adapttime,adapt_model,
                              adapter,init_x_num,incremental_x_num,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)
    intersect_num = len(intersected_df)
    print (res_dict)
    write_res(res_file,res_dict)

    # retrain a new model from scratch

    # build a new model
    new_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=int(other_params['max_depth']),
        learning_rate=other_params['learning_rate'],
        subsample=other_params['subsample'],
        colsample_bytree=other_params['colsample_bytree'],
        n_estimators=int(other_params['n_estimators']),
        min_child_weight=int(other_params['min_child_weight']),
    )


    start = time.perf_counter()
    new_model.fit(all_x_train,all_y_train)
    end = time.perf_counter()
    retraintime = end-start
    newmodelpath = list(modelpath)
    newmodelpath.insert(-4,'_retrain')
    newmodelpath = ''.join(newmodelpath)
    joblib.dump(new_model,newmodelpath)
    # 应用模型进行回归
    y_all_pred = np.abs(new_model.predict(x_all))
    res_dict = evaluate_model(y_all,y_all_pred)
    clf_regret,not_find,valdata_accuracy,new_data = index_regret_single_objeectie_Selector(label,new_model,intersected_df,all_x_train.columns)
    avg_regret = clf_regret/(intersect_num-not_find)

    res_dict = res_dict_write(res_dict,label,'new_all_data','new_'+modeltype,featuretype,sampling,feature_num,retraintime,new_model,
                              adapter,init_x_num,incremental_x_num,valdata_accuracy,all_reference_time,
                              clf_regret,intersect_num,avg_regret,not_find)

    print (res_dict)
    write_res(res_file,res_dict)


def run_systamatic_sampling(feature_type,reg_data_num,v_data_num):

    sampling = 'systematic_10thousand'
    # sampling = 'uniform_20thousand'
    # sampling = 'stratified_10thousand'
    pos = sampling.find('_')
    samplinglist = list(sampling)
    samplinglist.insert(pos,'_sampling')
    allsampling = ''.join(samplinglist)
    embeding_path = '/home/wamdm/chaohong/index_selection/embeding_'+sampling
    efile = 'has_ops_conv_'+allsampling+'_27795_embedings_'+str(reg_data_num)+'_all'

    embeding_file = os.path.join(embeding_path,efile)

    ratio_path = '/home/wamdm/chaohong/index_selection/ratio_data'
    ratio_file = os.path.join(ratio_path,sampling+'_total_feature_'+str(reg_data_num)+'_all')
    labels = ['throughput']
    for label in labels:

        # init_x_num = True_value_Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,
        #                                          v_data_num,reg_data_num,feature_type)
        init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,
                                      v_data_num,reg_data_num,feature_type)
    labels = ['bulkloadtime']
    for label in labels:
        # init_x_num = True_value_Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,
        #                                          v_data_num,reg_data_num,feature_type)
        init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,
                                      v_data_num,reg_data_num,feature_type)
    labels = ['indexsize']
    for label in labels:
        # init_x_num = True_value_Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,
        #                                          v_data_num,reg_data_num,feature_type)
        init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,
                                      v_data_num,reg_data_num,feature_type)

    # #
    # sampling = 'systematic_5thousand'
    # # sampling = 'uniform_20thousand'
    # # sampling = 'stratified_10thousand'
    # pos = sampling.find('_')
    # samplinglist = list(sampling)
    # samplinglist.insert(pos,'_sampling')
    # allsampling = ''.join(samplinglist)
    # embeding_path = '/home/wamdm/chaohong/index_selection/embeding_'+sampling
    # efile = 'has_ops_conv_'+allsampling+'_27795_embedings_'+str(reg_data_num)+'_all'
    #
    # embeding_file = os.path.join(embeding_path,efile)
    #
    # ratio_path = '/home/wamdm/chaohong/index_selection/ratio_data'
    # ratio_file = os.path.join(ratio_path,sampling+'_total_feature_'+str(reg_data_num)+'_all')
    # labels = ['throughput']
    # for label in labels:
    #
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    # labels = ['bulkloadtime']
    # for label in labels:
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    # labels = ['indexsize']
    # for label in labels:
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    #
    #
    # sampling = 'systematic_10thousand'
    # # sampling = 'uniform_20thousand'
    # # sampling = 'stratified_10thousand'
    # pos = sampling.find('_')
    # samplinglist = list(sampling)
    # samplinglist.insert(pos,'_sampling')
    # allsampling = ''.join(samplinglist)
    # embeding_path = '/home/wamdm/chaohong/index_selection/embeding_'+sampling
    # efile = 'has_ops_conv_'+allsampling+'_27795_embedings_'+str(reg_data_num)+'_all'
    #
    # embeding_file = os.path.join(embeding_path,efile)
    #
    # ratio_path = '/home/wamdm/chaohong/index_selection/ratio_data'
    # ratio_file = os.path.join(ratio_path,sampling+'_total_feature_'+str(reg_data_num)+'_all')
    # labels = ['throughput']
    # for label in labels:
    #
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    # labels = ['bulkloadtime']
    # for label in labels:
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    # labels = ['indexsize']
    # for label in labels:
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    #
    # sampling = 'systematic_15thousand'
    # # sampling = 'uniform_20thousand'
    # # sampling = 'stratified_10thousand'
    # pos = sampling.find('_')
    # samplinglist = list(sampling)
    # samplinglist.insert(pos,'_sampling')
    # allsampling = ''.join(samplinglist)
    # embeding_path = '/home/wamdm/chaohong/index_selection/embeding_'+sampling
    # efile = 'has_ops_conv_'+allsampling+'_27795_embedings_'+str(reg_data_num)+'_all'
    #
    # embeding_file = os.path.join(embeding_path,efile)
    #
    # ratio_path = '/home/wamdm/chaohong/index_selection/ratio_data'
    # ratio_file = os.path.join(ratio_path,sampling+'_total_feature_'+str(reg_data_num)+'_all')
    # labels = ['throughput']
    # for label in labels:
    #
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    # labels = ['bulkloadtime']
    # for label in labels:
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    # labels = ['indexsize']
    # for label in labels:
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    #
    # sampling = 'systematic_20thousand'
    # # sampling = 'uniform_20thousand'
    # # sampling = 'stratified_10thousand'
    # pos = sampling.find('_')
    # samplinglist = list(sampling)
    # samplinglist.insert(pos,'_sampling')
    # allsampling = ''.join(samplinglist)
    # embeding_path = '/home/wamdm/chaohong/index_selection/embeding_'+sampling
    # efile = 'has_ops_conv_'+allsampling+'_27795_embedings_'+str(reg_data_num)+'_all'
    #
    # embeding_file = os.path.join(embeding_path,efile)
    #
    # ratio_path = '/home/wamdm/chaohong/index_selection/ratio_data'
    # ratio_file = os.path.join(ratio_path,sampling+'_total_feature_'+str(reg_data_num)+'_all')
    # labels = ['throughput']
    # for label in labels:
    #
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    # labels = ['bulkloadtime']
    # for label in labels:
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)
    # labels = ['indexsize']
    # for label in labels:
    #     init_x_num = Hybrid_XGBRessor(label,embeding_file,ratio_file,sampling,allsampling,v_data_num,reg_data_num,feature_type)



def run_XGB(reg_data_num,v_data_num):
    print ('Lord, thank You be with chao! chao needs You from ever to ever')
    print ('Lord, we are begin to run XGB')

    feature_type = "hybridencoder"

    run_systamatic_sampling(feature_type,reg_data_num,v_data_num)

def run_RF(reg_data_num,v_data_num):
    print ('Lord, thank You be with chao! chao needs You from ever to ever')
    print ('Lord, we are begin to run RF')


    for num in range(4):
        feature_type = "hybridencoder"
        sampling = 'systematic_10thousand'
        # sampling = 'uniform_20thousand'
        # sampling = 'stratified_1thousand'
        pos = sampling.find('_')
        samplinglist = list(sampling)
        samplinglist.insert(pos,'_sampling')
        allsampling = ''.join(samplinglist)
        embeding_path = '/home/wamdm/chaohong/index_selection/embeding_'+sampling
        efile = 'has_ops_conv_'+allsampling+'_27795_embedings_'+str(reg_data_num)+'_all'
        embeding_file = os.path.join(embeding_path,efile)

        ratio_path = '/home/wamdm/chaohong/index_selection/ratio_data'
        ratio_file = os.path.join(ratio_path,sampling+'_total_feature_'+str(reg_data_num)+'_all')

        labels = ['indexsize']
        for label in labels:
            init_x_num = Hybrid_RFRegressor(label,embeding_file,ratio_file,sampling,allsampling,
                                            v_data_num,reg_data_num,feature_type)
        labels = ['bulkloadtime']
        for label in labels:
            init_x_num = Hybrid_RFRegressor(label,embeding_file,ratio_file,sampling,allsampling,
                                            v_data_num,reg_data_num,feature_type)
        labels = ['throughput']
        for label in labels:
            init_x_num = Hybrid_RFRegressor(label,embeding_file,ratio_file,sampling,allsampling,
                                            v_data_num,reg_data_num,feature_type)

    #
    # adapter = 'libio'
    # labels = ['indexsize']
    # for label in labels:
    #     modelpath,parapath,init_x_num = Hybrid_RFRegressor(label,embeding_file,sampling,feature_type)
    #     modelpath,parapath,init_x_num = Hybrid_RFRegressor(label,embeding_file,sampling,feature_type,adapter=adapter)
    #     incremental_single_RF(label,modelpath,parapath,adapter,feature_type,sampling,'singleRF')
    # labels = ['bulkloadtime']
    # for label in labels:
    #     modelpath,parapath,init_x_num = Hybrid_RFRegressor(label,embeding_file,sampling,feature_type)
    #     modelpath,parapath,init_x_num = Hybrid_RFRegressor(label,embeding_file,sampling,feature_type,adapter=adapter)
    #     incremental_single_RF(label,modelpath,parapath,adapter,feature_type,sampling,'singleRF')
    #
    # adapter = '25rq'
    # labels = ['indexsize']
    # for label in labels:
    #     modelpath,parapath,init_x_num = Hybrid_RFRegressor(label,embeding_file,sampling,feature_type)
    #     modelpath,parapath,init_x_num = Hybrid_RFRegressor(label,embeding_file,sampling,feature_type,adapter=adapter)
    #     incremental_single_RF(label,modelpath,parapath,adapter,feature_type,sampling,'singleRF')
    # labels = ['bulkloadtime']
    # for label in labels:
    #     modelpath,parapath,init_x_num = Hybrid_RFRegressor(label,embeding_file,sampling,feature_type)
    #     modelpath,parapath,init_x_num = Hybrid_RFRegressor(label,embeding_file,sampling,feature_type,adapter=adapter)
    #     incremental_single_RF(label,modelpath,parapath,adapter,feature_type,sampling,'singleRF')


if __name__ == '__main__':
    reg_data_num = 28008
    v_data_num = 29851
    run_RF(reg_data_num,v_data_num)
    for i in range (1):   # 感谢天主，测试通了
        run_XGB(reg_data_num,v_data_num)

