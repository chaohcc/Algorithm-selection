import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import os,csv
from torch.autograd import Variable
import random


# 用dwmatrix 作为label，train model in a self-supervised manner

# Setting random seed to facilitate reproduction
seed = 9958
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# cuda environment is recommened
device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
print(device)

# Model Structure

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.4):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.4):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):

        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, y, x):
        for layer in self.layers:
            x = layer(y, x)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, y, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(y, x, x))

        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, y, x):
        for layer in self.layers:
            x = layer(y, x)
        return self.norm(x)


# Pooling by attention
class PoolingLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(PoolingLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, y, x):
        x = self.self_attn(y, x, x)
        return self.sublayer(x, self.feed_forward)

def make_model(d_model, N, d_ff, h, dropout=0.4):   #d_model: embeding size, N:n_encoder_layers,d_ff: dim3 (128), h:n_heads(8)

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    encoder_model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    encoder_pooling_model = Encoder(PoolingLayer(d_model, c(attn), c(ff), dropout), 1)
    decoder_model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    decoder_pooling_model = Encoder(PoolingLayer(d_model, c(attn), c(ff), dropout), 1)

    return encoder_model, encoder_pooling_model,decoder_model,decoder_pooling_model

# Completed encoder
class self_attn_encoder_model(nn.Module):

    def __init__(self, encoder, encoder_pooling_model, ini_feats, encode_feats,channel_feats):
        super(self_attn_encoder_model, self).__init__()
        c = copy.deepcopy
        self.encoder = encoder
        self.conv2d1 = nn.Conv2d(1,channel_feats,(2, 1),(2,1))
        # self.conv2d2 = nn.Conv2d(256,128,(4, 1),(2,1))
        self.conv2d3 = nn.Conv2d(channel_feats,1,(2, 1),(2,1))
        self.maxpool = nn.MaxPool2d((2,1),stride=(2,1),return_indices=True)
        # self.encode_pma1 = c(encoder_pooling_model)
        self.encode_pma2 = c(encoder_pooling_model)
        self.S = nn.Parameter(torch.Tensor(1,1,encode_feats))
        nn.init.xavier_uniform_(self.S)
        self.linear1 = nn.Linear(ini_feats, encode_feats, bias = True)

    def forward(self, batch_samples):
        # 开始时用 maxpooling in CNN
        # 开始时不用relu 和 linear
        batch_samples = batch_samples.unsqueeze(1)  #1*20000*32
        batch_samples = self.conv2d1(batch_samples)  # 256*9999*32
        batch_samples,indices = self.maxpool(batch_samples)
        batch_samples = F.relu(batch_samples)
        batch_samples = self.conv2d3(batch_samples)  #  1*2499 *32
        conv_size = batch_samples.size(2)
        batch_samples = batch_samples.squeeze(1)  # 3*4999*32
        batch_samples = self.linear1(batch_samples)  # 20000*25  -- 20000*32
        # attn_output = self.encode_pma1(self.S.repeat(batch_samples.size(0),1000,1), batch_samples)  #1000*32
        # attn_output = self.encoder(attn_output, attn_output)   # 1000*32

        batch_samples = self.encoder(batch_samples, batch_samples)   # 1000*32

        batch_samples = self.encode_pma2(self.S.repeat(batch_samples.size(0),1,1), batch_samples)  #1000*32

        return batch_samples, conv_size,indices

# Completed decoder
class self_attn_decoder_model(nn.Module):
    def __init__(self, decoder, decoder_pooling_model, ini_feats, encode_feats,channel_feats):
        super(self_attn_decoder_model, self).__init__()
        c = copy.deepcopy
        self.decoder = decoder
        self.decode_pma1 = c(decoder_pooling_model)
        self.decode_pma2 = c(decoder_pooling_model)
        # self.decode_pma3 = c(decoder_pooling_model)
        self.S = nn.Parameter(torch.Tensor(1,1,encode_feats))
        nn.init.xavier_uniform_(self.S)
        self.output1 = nn.Linear(encode_feats, ini_feats, bias = True)
        self.convtrans1 = nn.ConvTranspose2d(channel_feats,1,(2, 1),(2,1))
        self.convtrans3 = nn.ConvTranspose2d(1, channel_feats,(2, 1),(2,1))
        self.unmaxpool = nn.MaxUnpool2d((2,1),(2,1))

    def forward(self, attn_output,out_feats,indices):
        # 开始时用 maxpooling in CNN
        # 开始时不用relu 和 linear
        # beginning dimension 1*32

        # back_output = self.decode_pma2(self.S.repeat(attn_output.size(0),1000,1), attn_output)   #1000*32
        attn_output = self.decode_pma2(self.S.repeat(attn_output.size(0),out_feats,1), attn_output)   #1000*32
        attn_output = self.decoder(attn_output, attn_output)   #1000*32
        # back_output = self.decode_pma1(self.S.repeat(back_output.size(0),out_feats,1), back_output)   #4999*32

        attn_output = self.output1(attn_output)
        attn_output = attn_output.unsqueeze(1)  # 1*2499*32
        attn_output = self.convtrans3(attn_output)   # 1*2499*32
        attn_output = F.relu(attn_output)
        attn_output = self.unmaxpool(attn_output,indices)
        attn_output = self.convtrans1(attn_output)  #1*20000*32
        attn_output = attn_output.squeeze(1)

        return attn_output



class self_attn_model(nn.Module):

    def __init__(self, attn_encoder, attn_decoder):
        super(self_attn_model, self).__init__()
        c = copy.deepcopy
        self.attn_encoder = attn_encoder
        self.attn_decoder = attn_decoder

    def forward(self, batch_samples):
        # 开始时用 maxpooling in CNN

        attn_output,convsize,indices = self.attn_encoder(batch_samples)
        back_output = self.attn_decoder(attn_output,convsize,indices)

        return attn_output,back_output



def retrun_embeding(model,data,device,bs):
    model.eval()
    embdedings = []
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i in range(0,len(data),bs):
            iinputs = data[i:i+bs].to(device)
            iinputs = iinputs.to(device).float()

            attn_output = model(iinputs)[0].detach().cpu().squeeze(dim=1).numpy()
            embdedings.extend(attn_output.tolist())
            iinputs = None
            del iinputs
    data = None
    del data
    return embdedings


def load_ops_data(ops_list_path,dwfilepath, test = False):

    # Load LIAS dataset
    #只读取 有ops 文件，list 是total_feature_has_ops
    # dwfilepath = '/data/xxx/dw_matrix_span_onehot/'
    # dwfilepath = '/data/xxx/30thousand_stratified_dw_matrix_span_onehot/'  # 20thousand
    # dwfilepath = '/data/xxx/10thousand_stratified_dw_matrix_span_onehot/'   # 10thousand
    # dwfilepath = '/data/xxx/40thousand_stratified_dw_matrix_span_onehot/'   # 40thousand
    # dwfilepath = '/data/xxx/uniform_30thousand_dw_matrix_span_onehot/'   # uniform 20thousand
    # dwfilepath = '/data/xxx/systematic_1thousand_dw_matrix_span_onehot/'   # systematic 20thousand

    print(dwfilepath)
    indata = pd.read_csv(ops_list_path)

    has_ops_list = indata[['dataname','opsname']]

    data=[]
    cut_i = 0
    # 根据data name 和ops name，构建DW_matrix
    # 如果已经有DW_matrix 则直接读入: DW_matrix 的名称为dataname+_ops_name(prefix)
    # 根据data name 读取pgm feature, 根据ops name 读取operations file
    # 2 代表insert, 1 代表range query, 0 代表 point query
    ci = 0
    lastfile = None
    for dataname,opsname in has_ops_list.values:
        ops_prefix = opsname.split('_')[0]
        dwfile = os.path.join(dwfilepath,dataname+'_'+ops_prefix)
        if dwfile == lastfile:
            # newline = lines+index_library[has_ops_indexes.loc[i]]
            # cut_i += 1
            # if cut_i >= 5:
            #     continue
            data.append(lines)
        else:
            cut_i = 0
            if (os.path.exists(dwfile)):
                lastfile = dwfile
                lines = pd.read_csv(dwfile,header=None).values.tolist()
                # newline = lines+index_library[has_ops_indexes.loc[i]]
                data.append(lines)
            else:
                print('Lord, chao needs You! dwfile does not exists~',dwfile)

        ci += 1

        if test and ci == 100:
            break
    print('Lord, thank You! ')
    # Load dataset

    len_test = len(data)
    print(f'Size of test dataset: {len_test}')

    return data, has_ops_list, len_test

def embed_load_ops_data(ops_list_path,dwfilepath, test = False):

    # Load LIAS dataset
    #只读取 有ops 文件，list 是total_feature_has_ops
    # dwfilepath = '/data/xxx/dw_matrix_span_onehot/'
    # dwfilepath = '/data/xxx/30thousand_stratified_dw_matrix_span_onehot/'  # 20thousand
    # dwfilepath = '/data/xxx/10thousand_stratified_dw_matrix_span_onehot/'   # 10thousand
    # dwfilepath = '/data/xxx/40thousand_stratified_dw_matrix_span_onehot/'   # 40thousand
    # dwfilepath = '/data/xxx/uniform_30thousand_dw_matrix_span_onehot/'   # uniform 20thousand
    # dwfilepath = '/data/xxx/systematic_1thousand_dw_matrix_span_onehot/'   # systematic 20thousand

    print(dwfilepath)
    indata = pd.read_csv(ops_list_path)

    has_ops_list = indata[['dataname','opsname']]
    unique_data_ops = has_ops_list.drop_duplicates(subset=['dataname','opsname'])
    unique_data_ops = unique_data_ops.reset_index(drop=True)

    data=[]
    cut_i = 0
    # 根据data name 和ops name，构建DW_matrix
    # 如果已经有DW_matrix 则直接读入: DW_matrix 的名称为dataname+_ops_name(prefix)
    # 根据data name 读取pgm feature, 根据ops name 读取operations file
    # 2 代表insert, 1 代表range query, 0 代表 point query
    ci = 0
    lastfile = None
    for dataname,opsname in unique_data_ops.values:
        ops_prefix = opsname.split('_')[0]
        dwfile = os.path.join(dwfilepath,dataname+'_'+ops_prefix)
        if dwfile == lastfile:
            # newline = lines+index_library[has_ops_indexes.loc[i]]
            # cut_i += 1
            # if cut_i >= 5:
            #     continue
            data.append(lines)
        else:
            cut_i = 0
            if (os.path.exists(dwfile)):
                lastfile = dwfile
                lines = pd.read_csv(dwfile,header=None).values.tolist()
                # newline = lines+index_library[has_ops_indexes.loc[i]]
                data.append(lines)
            else:
                print('Lord, chao needs You! dwfile does not exists~',dwfile)

        ci += 1

        if test and ci == 100:
            break
    print('Lord, thank You! ')
    # Load dataset

    len_test = len(data)
    print(f'Size of unique data ops: {len_test}')
    total_len = len(has_ops_list)
    print(f'Size of reg data: {total_len}')

    return data, unique_data_ops, len_test, has_ops_list,total_len


def train_model_4_embedding(ops_list_path,dwfilepath,num,bsize):

    # data, has_ops_list, len_test = load_ops_data(ops_list_path,dwfilepath,True)
    data, has_ops_list, len_test = load_ops_data(ops_list_path,dwfilepath,False)
    has_ops_list = None
    del has_ops_list
    # model Inference
    dim1 = 32 # embedding size
    dim2_1 = 256 # hidden dimension 1 for prediction layer
    dim2_2 = 128 # hidden dimension 2 for prediction layer
    dim2_3 = 64
    dim2_4 = 32
    dim3 = 128 # hidden dimension for FNN

    n_encoder_layers = 3        # number of layer of attention encoder
    n_heads = 8 # number of heads in attention
    dropout_r = 0.2 # dropout ratio
    bs = bsize # batch size
    EPOCHS = 60
    learn_rate = 0.001
    channel_dim = 128
    print('n_encoder_layers: ',n_encoder_layers)
    encoder_model, encoder_pooling_model,decoder_model,decoder_pooling_model = make_model(dim1, n_encoder_layers, dim3, n_heads, dropout=dropout_r)
    attn_encoder = self_attn_encoder_model(encoder_model, encoder_pooling_model, 25, dim1,channel_dim)
    attn_decoder = self_attn_decoder_model(decoder_model,decoder_pooling_model, 25, dim1,channel_dim)

    model = self_attn_model(attn_encoder, attn_decoder)
    # para_dict_loc = './model/LIB_config.pth'
    # model.load_state_dict(torch.load(para_dict_loc))
    # print(model)
    device_ids = [1,0,2]
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-06)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    loss_func = nn.MSELoss()

    # random.shuffle(data)
    # data = data[:15000]
    data = torch.tensor(data)
    trainloader = torch.utils.data.DataLoader(dataset=data, batch_size=bs, shuffle=True)
    data = None
    del data
    # Model Evaluation on test data
    def train_encoder_model(model, EPOCHS):
        para_path = ''
        min_loss = 90000000000000
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            running_loss = 0.0
            for i,iinputs in enumerate(trainloader):
                optimizer.zero_grad()
                iinputs = iinputs.float()
                iinputs = Variable(iinputs).to(device)
                outputs = model(iinputs)[1].squeeze(dim=1)
                loss = loss_func(outputs, iinputs)

                # ============ Backward ============
                loss.backward()
                optimizer.step()
                iinputs = None
                del iinputs
                torch.cuda.empty_cache()
                # ============ Logging ============
                running_loss += loss.data
                epoch_loss += loss.data/1000000000
                if i % 500 == 0:
                    print('[%d, %5d] loss: %.8f' %
                          (epoch + 1, i + 1, running_loss/500 ))
                    running_loss = 0
                    print('Lord, thank You! You are my refuge from ever to ever ~ ')
            if (epoch_loss < min_loss):
                min_loss = epoch_loss
                para_path = './'+str(num)+'_run2_transmodel_para_conv_surpervised_'+ str(len_test)+'_inputdim_'+str(32) +'_stratified_sampling_'+str(epoch)+'.pth'
                torch.save(model.state_dict(),para_path)
                print ('Lord, thank You, we have saved the model to ', para_path)
                print('Lord, the min loss is ', min_loss)
            epoch_loss = 0
            # scheduler.step()
            print ("Lord, thank You! the learning rate is: ", optimizer.state_dict()['param_groups'][0]['lr'])
        return para_path
    print('Lord, we begin to train the model')
    start = time.perf_counter()
    para_path = train_encoder_model(model,EPOCHS)
    end = time.perf_counter()
    traintime = end-start
    trainloader = None
    del trainloader
    print ('Lord, thank You, the model train over~')

    # 只保存模型参数
    # para_path = './transmodel_para_conv_surpervised_'+ str(len_test)+'_inputdim_'+str(32) +'_stratified_sampling'+'.pth'
    # torch.save(model.state_dict(),para_path)
    # print ('Lord, thank You, we have saved the model to ', para_path)
    # 加载最优模型参数
    print(para_path)
    model.load_state_dict(torch.load(para_path))
    data, has_ops_list, len_test = load_ops_data(ops_list_path,dwfilepath,True)
    total_embeding = []
    big_batch = 10000
    for i in range(0,len_test,big_batch):

        batch_data = data[i:i+big_batch]
        batch_data = torch.tensor(batch_data)

        print ('Lord, thank, we are beginning to get the embedings')
        embedings = retrun_embeding(model,batch_data,device,bs)
        total_embeding.extend(embedings)
        print ('Jesus, please has pity on chao! thank You~')
    # 把embedings 保存下来
    embed_head = []
    for i in range(32):
        embed_head.append('em'+str(i))
    total_embeding = pd.DataFrame(total_embeding,columns= embed_head)
    has_ops_list = pd.concat([has_ops_list,total_embeding],axis= 1)

    embeding_file = '/data/xxx/run2_has_ops_conv_surpervised_embedings_'+str(len_test)+'_stratified_sampling'
    with open(embeding_file, 'w') as ef:
        efwriter = csv.writer(ef)
        efwriter.writerow(list(has_ops_list.columns))
        efwriter.writerows(has_ops_list.values)
    print ('Lord, thank You, the optimal model is: ', para_path)
    print ('Lord, please be with Long! the embeding file is ',embeding_file)
    return len_test,traintime



def batch_load_model_4_embedding(para_path,model_len,ops_flags,dwfilepath,sampling):

    dim1 = 32 # embedding size
    dim2_1 = 256 # hidden dimension 1 for prediction layer
    dim2_2 = 128 # hidden dimension 2 for prediction layer
    dim2_3 = 64
    dim2_4 = 32
    dim3 = 128 # hidden dimension for FNN
    n_encoder_layers = 3        # number of layer of attention encoder
    n_heads = 8 # number of heads in attention
    dropout_r = 0.2 # dropout ratio
    bs = 32 # batch size
    EPOCHS = 20
    learn_rate = 0.001
    channel_dim = 128
    print('n_encoder_layers: ',n_encoder_layers)
    encoder_model, encoder_pooling_model,decoder_model,decoder_pooling_model = make_model(dim1, n_encoder_layers, dim3, n_heads, dropout=dropout_r)
    attn_encoder = self_attn_encoder_model(encoder_model, encoder_pooling_model, 25, dim1,channel_dim)
    attn_decoder = self_attn_decoder_model(decoder_model,decoder_pooling_model, 25, dim1,channel_dim)

    load_model = self_attn_model(attn_encoder, attn_decoder)

    # 加载模型
    device_ids = [1,0,2]
    load_model = nn.DataParallel(load_model, device_ids=device_ids)
    load_model.load_state_dict(torch.load(para_path))
    load_model = load_model.to(device)

    for ops_flag in ops_flags:
        ops_list_path = ops_flag[0]
        flag = ops_flag[1]

        # load data
        # data, unique_data_ops, len_test, has_ops_list,total_len = embed_load_ops_data(ops_list_path,dwfilepath,True)
        data, unique_data_ops, len_test, has_ops_list,total_len = embed_load_ops_data(ops_list_path,dwfilepath)

        total_embeding = []
        big_batch = 5000
        for i in range(0,len_test,big_batch):

            batch_data = data[i:i+big_batch]
            batch_data = torch.tensor(batch_data)
            print ('Lord, thank You, we are beginning to get the embedings')
            embedings = retrun_embeding(load_model,batch_data,device,bs)

            total_embeding.extend(embedings)
            print ('Jesus, please has pity on chao! thank You~')

        # 把embedings 保存下来
        embed_head = []
        for i in range(32):
            embed_head.append('em'+str(i))
        total_embeding = pd.DataFrame(total_embeding,columns= embed_head)
        unique_embeding = pd.concat([unique_data_ops,total_embeding],axis= 1)
        emsl=[]

        for index, row in has_ops_list.iterrows():
            dd=unique_embeding[(unique_embeding['dataname']==row['dataname']) & (unique_embeding['opsname']==row['opsname'])]
            ems=dd[['em%s'%m for m in range(32)]]
            emsl.append(ems)
        datam=pd.concat(emsl,axis=0)
        merge_embeding=pd.concat([has_ops_list,datam.reset_index(drop=True)],axis=1)
        # embeding_file = '/data/xxx/has_ops_conv_stratified_sampling_10thousand_'+str(model_len)+'_embedings_'+str(len_test)
        embeding_file = '/data/xxx/has_ops_conv'+sampling+str(model_len)+'_embedings_'+str(total_len)
        # embeding_file = '/data/xxx/has_ops_conv_systematic_sampling_5thousand_'+str(model_len)+'_embedings_'+str(len_test)

        if flag:
            embeding_file += flag
        with open(embeding_file, 'w') as ef:
            efwriter = csv.writer(ef)
            efwriter.writerow(list(merge_embeding.columns))
            efwriter.writerows(merge_embeding.values)

        print ('Lord, thank You, the embeding file has been written to ~')
        print(embeding_file)


if __name__ == "__main__":
    print('Lord, You are my refuge!')

    ops_list_path = '/data/xxx/total_feature/total_feature_20thousand_systematic_has_ops_12_27795'

    bs = 96 # batch size used to train the model

    dwfilepath = ['Your path of DW_Matrix']   # systematic 20thousand

    # optional:  train model
    model_len,traintime = train_model_4_embedding(ops_list_path,dwfilepath,3,bs)
    print(dwfilepath)
    print ('train time is: ', traintime)


    print('Lord, thank You, we are beginning to load model get embedding')

    ops_list_path1 = 'the list file to be encoded'

    ops_files = [ops_list_path1]

    flags = ['any name you want related to ops file']

    ops_flags = zip(ops_files,flags)

    para_path = 'model parameter path'

    print(para_path)
    sampling = '_systematic_sampling_10thousand_'

    batch_load_model_4_embedding (para_path,model_len, ops_flags,dwfilepath,sampling)


    print ('O~Lord, You are my refuge, You are always with chao')
