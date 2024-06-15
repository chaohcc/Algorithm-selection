# LINDAS
LINDAS: a Learned Approach to Index Algorithm Selection

LINDAS aims to select an index algorithm for a specific dataset and workload. 
## Some images may do not Some images may not display properly. For more detail, you can download the file: "full_version_technical_report_a_learned_approach_to_index_algorithm_selection.pdf".
---

### performance measures

Considered performance measures in LINDAS: bulk load time, index size, throughput. The correlations among the three metrics

![task_corr](https://github.com/chaohcc/LIAS/assets/51820918/cc02b9b1-f063-47cb-a0db-ef03c8f0240b)

---

### Environment requirement
* Python 3.8
* xgboost 2.0.0
* torch 1.7.1+cu110
* scikit-learn 1.3.1

---

### training data description
#### The training data consist of the performance evaluation of 13 index algorithms on 12 datasets (Table 1) with various workloads (Table 2) described in the paper.
#### The format of the training data: each row can be seen as a certain performance evaluation on a specific dataset and workload, with a specific index algorithm.
#### The training data are stored in csv format. You can use data = pd.read_csv(filename) to load the data. This example contains 6,000+ records. The total number of training data is 28,000 will be published when the paper is published.

The features in each row.

dataname,

DV:datasize,max_key,minkey,pwl_num4,max_size4,min_size4,max_gap4,min_gap4,pwl_num8,max_size8,min_size8,max_gap8,min_gap8,pwl_num16,max_size16,min_size16,max_gap16,min_gap16,pwl_num32,max_size32,min_size32,max_gap32,min_gap32,pwl_num64,max_size64,min_size64,max_gap64,min_gap64,pwl_num128,max_size128,min_size128,max_gap128,min_gap128,pwl_num256,max_size256,min_size256,max_gap256,min_gap256,pwl_num512,max_size512,min_size512,max_gap512,min_gap512,pwl_num1024,max_size1024,min_size1024,max_gap1024,min_gap1024,pwl_num2048,max_size2048,min_size2048,max_gap2048,min_gap2048,pwl_num4096,max_size4096,min_size4096,max_gap4096,min_gap4096,

workload_feature_1: ops_num,rq,nl,i,m,hotratio,thread,

index_feature: xf0,xf1,xf2,xf3,xf4,xf5,xf6,xf7,xf8,xf9,xf10,xf11,xf12,xf13,xf14,xf15,xf16,xf17,xf18,xf19,xf20,xf21,xf22,xf23,xf24,xf25,xf26,xf27,xf28,xf29,xf30,xf31,xf32,xf33,xf34,xf35,xf36,xf37,xf38,xf39,xf40,xf41,xf42,

opsname,indexname,

performance_metrics: bulkloadtime,indexsize,throughput,

workload_feature_2: em0,em1,em2,em3,em4,em5,em6,em7,em8,em9,em10,em11,em12,em13,em14,em15,em16,em17,em18,em19,em20,em21,em22,em23,em24,em25,em26,em27,em28,em29,em30,em31

---

### description for the dataset used in training data collection [GRE]


#### <center>Tabel 1:  dataset description</center>
| dataset | description | duplicate | reference |
| :---:     |     :---:      |       :---:  |       :---:  |
books    | Amazon book sales popularity                                                                                    | No       | SOSD      |
fb        | Upsampled Facebook user ID                                                                                                                                 | No        |   SOSD     |
osm       | Uniformly sampled OpenStreetMap locations                                                                                                                  | No        |   SOSD    |
wiki      | Wikipedia article edit timestamps                                                                                                                          | Yes       |    SOSD    |
covid     | Uniformly sampled Tweet ID with tag COVID-19                                                                                                               | No        |    SNAM   |
genome    | Loci pairs in human chromosomes                  | No        |    Cell    |
stack     | Vote ID from Stackoverflow                                                                                                                                 | No       |     Stackoverflow   |
wise      | Partition key from the WISE data                                                                                                                           | No       |     AJ   |
libio     | Repository ID from libraries.io                                                                                                                            | No        |  Libraries.io.    |
history   | History node ID in OpenStreetMap                                                                                                                           | No        |    OpenStreetMap    |
planet    | Planet ID in OpenStreetMap                                                                                                                                 | No        |   OpenStreetMap    |
lognormal | Values generated according to a lognormal distribution, multiply $10^9$ and rounded down to the nearest integer | Yes       |    ALEX    | 

---

### other files
#### slef_Conv_supervised_antoencoder.py
the python file for training the workload encoder model.
#### epoch64_bs_96_thansmodel_conv_supervised_27795_inputdim_32_systematic_sampling_10thousand_encoder.pth
the saved model of workload encoder, trained on training data with 277775 training points.
the workload file are sampling with systematic method, that sampling 10000 operations from each workload file
the batch size if 96, trian with 64 epoches.
#### LINDAS-clf, LINDAS-reg
files for the classfication model and regression model
#### DWMatrix
some examples for the DWMatrix for the workload file
#### training_data
contains training data for the LINDAS-clf and LINDAS-reg

### References
[GRE] Chaichon Wongkham, Baotong Lu, Chris Liu, Zhicong Zhong, Eric Lo, and Tianzheng Wang. Are Updatable Learned Indexes Ready? . PVLDB,15(11): 3004 - 3017, 2022.

[SOSD] Andreas Kipf, Ryan Marcus, Alexander van Renen, Mihail Stoian, Alfons Kemper,Tim Kraska, and Thomas Neumann. 2019. SOSD: A Benchmark for Learned Indexes. NeurIPS Workshop on Machine Learning for Systems (2019).

[SNAM] Christian E Lopez and Caleb Gallemore. 2021. An augmented multilingual Twitter dataset for studying the COVID-19 infodemic. Social Network Analysis and Mining 11, 1 (2021), 1–14.

[Cell] Suhas S.P. Rao and et al. 2014. A 3D Map of the Human Genome at Kilobase Resolution Reveals Principles of Chromatin Looping. Cell (2014).

[Stackoverflow] Stackoverflow. 2021. Vote ID. (2021). https://archive.org/download/stackexchange.

[Aj] Edward L Wright, Peter RM Eisenhardt, Amy K Mainzer, Michael E Ressler, Roc M Cutri, Thomas Jarrett, J Davy Kirkpatrick, Deborah Padgett, Robert S McMillan, Michael Skrutskie, et al. 2010. The Wide-field Infrared Survey Explore (WISE): mission description and initial on-orbit performance. The Astronomical Journal 140, 6 (2010), 1868.

[Libraries.io.] Libraries.io. 2017. Repository ID. (2017). https://libraries.io/data. 

[OpenStreetMap] Google Cloud. 2017. OpenStreetMap. (2017). https://console.cloud.google.com/marketplace/details/openstreetmap/geo-openstreetmap

[ALEX] Jialin Ding, Umar Farooq Minhas, Jia Yu, Chi Wang, Jaeyoung Do, Yinan Li,Hantian Zhang, Badrish Chandramouli, Johannes Gehrke, Donald Kossmann, et al. 2020. ALEX: an updatable adaptive learned index. Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data, 969–984.

## Some experimental results

### 1. LINDAS-reg: multiple constraints

|optimize on| avg\_regret| constraints on   |   
|--   | :---:|:---:|
|throughput   |0.028   |{[}index size, bulk load time{]} |
|index size    | 0.072     |{[}throughput, bulk load time{]} |
|bulk load time|0.068      |{[}throughput, index size{]}   |

### 2. time efficiency (seconds)

|dataset     | wise   |genome}   |
|--   | :---:|:---:|
data size ({M})| 50    & 100   & 150  & 200  | 50  & 100   & 150   & 200  |
LINDAS    |    13.21 & 26.07 & 39.2 & 52.22 | 13.94 & 28.48 & 44.07 & 60.0 |
Exhaustive   | 765.3  1063   1545 & 1927 | 798.0 & 1118 &  1589 & 3606 |

### The model size and training time for different models
![image](https://github.com/chaohcc/Algorithm-selection/assets/51820918/33a7f06b-7feb-49f5-962a-14779ccb13bc)

### ablation study on LINDAS-reg on dataset and workload features, as well as index features: measured with regret

|features | #_features |  index size | bulkload time | throughput |
|--   | :---:|:---:|:---:|:---:|
|DH + w/o DW\_Matrix | 45                              |0.90        | 0.59          | 0.87       |
|DH + DW\_Matrix|      84                              | 0.008      | 0.034        | 0.028     |
|DV + w/o DW\_Matrix | 101              |0.76      | 0.383         | 0.763      |
|DV +  DW\_Matrix     | 140                |\textbf{0.002}      | \textbf{0.03}      | \textbf{0.016}   |

|features |#\_features  |index size |    bulkload time   | throughput      |
|--   | :---:|:---:|:---:|:---:|
| w/o XF             |98           | 0.513          | 0.741      | 0.189      |
| XF         |140                      | \textbf{0.002} | \textbf{0.03} | \textbf{0.016} |


