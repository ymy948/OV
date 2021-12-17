# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:13:10 2021

@author: DELL
"""
import pandas as pd
import numpy as np
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

## 461样本 无normal 存在Recurrent Solid Tumor 8个 剩余样本453个
m = pd.read_csv('E:\sirebrowser\OV\miRNA\OV.miRseq_mature_RPM.csv',engine='python')
a = m.values 
a = list(map(list, zip(*a))) 
a = pd.DataFrame(a) 
a.iloc[0:2,0:2]
a.insert(0,'sample',m.columns)
a.columns = a.iloc[0,:]
b = a.iloc[1:,:]
b.iloc[0:2,0:2]
b.rename(columns={'锘縂ene':'Gene'}, inplace=True) 
b = b.set_index('Gene',drop=False)
b['Gene'] = b['Gene'].apply(lambda x:x[:12]).tolist()
b = b[~b.Gene.duplicated()]
b.info()
## 筛选578中453个
c = pd.read_csv('E:\sirebrowser\OV\clinical\clin_info.csv',engine='python')
cc = c.set_index('sample',drop=False)
for i in range(0,452): # 根据报错修改数值577 452个有clin_info
    if list(cc['sample'])[i] in list(b['Gene']):
        continue
    else:
        cc.drop(index=[list(cc['sample'])[i]],inplace=True)
for i in range(0,452): 
    if list(b['Gene'])[i] in list(cc['sample']):
        continue
    else:
        b.drop(index=[list(b['Gene'])[i]],inplace=True)
cc.to_csv('E:\\sirebrowser\\OV\\miRNA\\452clin_info.csv',index=0)
b.to_csv('E:\\sirebrowser\\OV\\mRNA\\452samples+2588miRNAs.csv',index=0)


## 删除空值大于样本值20%的gene
452*0.2 
nan = b.isnull().sum(axis=0)
b.drop(columns=nan[nan >= 90].index,inplace=True)
b.info()  # 388miRNA

# 删除空值大于gene20%的样本
388*0.2 
nan = b.isnull().sum(axis=1)
nan[nan>77] # 无样本
b.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs.csv',index=0)

## knn填补-1
import sklearn
import sklearn.impute    
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score as CVS
import matplotlib.pyplot as plt
b = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs.csv',engine='python')
t2 = b.drop('Gene',axis=1)
t2.iloc[0:2,0:2]
nan = t2.isnull().sum(axis=1)
nan[nan==0] # 无空值样本有18个
imputer = KNNImputer(n_neighbors=5,weights='distance') 
imputed = imputer.fit_transform(t2) 
df_imputed = pd.DataFrame(imputed, columns=t2.columns) # 此时无行标题
df_imputed.iloc[0:2,0:2]
# b = b.set_index(df_imputed.index)
df_imputed.insert(0,'Gene',b['Gene'])
type(df_imputed)
df_imputed.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_knn1.csv',index=0)
df_imputed.isnull().sum(axis=0)

## 分位数归一化
# 转置
data = df_imputed.values 
index1 = list(df_imputed.keys()) 
data = list(map(list, zip(*data))) 
data = pd.DataFrame(data, index=index1) 
data.iloc[0:2,0:2]
data.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_knn1_T.csv', header=0)
# 处理
rpm_no_nan_t = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_knn1_T.csv',engine='python')
rpm_no_nan_t.iloc[0:2,0:2]
rpm_no_nan_t_2 = rpm_no_nan_t.drop('Gene',axis=1)
rpm_no_nan_t_2.iloc[0:2,0:2]

rank_mean = rpm_no_nan_t_2.stack().groupby(rpm_no_nan_t_2.rank(method='first').stack().astype(int)).mean()
rpm_qn = rpm_no_nan_t_2.rank(method='min').stack().astype(int).map(rank_mean).unstack()
rpm_qn.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_qn.csv',index=0)
#写入列名称
rpm_qn = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_qn.csv',engine='python')
rpm_qn.iloc[0:2,0:2]
rpm_qn.insert(0,'Gene',rpm_no_nan_t['Gene']) # 有小问题 通过excel修改
rpm_qn.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_qn.csv',index=0)

## knn-2
# 找出knn-1之前的空值并删除
rpm_t = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs.csv',engine='python') # knn-1之前的数据
rpm_t.iloc[0:3,0:4]
#转置 
data = rpm_t.values # data是数组，直接从文件读出来的数据格式是数组
index1 = list(rpm_t.keys()) # 获取原有csv文件的标题，并形成列表
data = list(map(list, zip(*data))) # map()可以单独列出列表，将数组转换成列表
data = pd.DataFrame(data, index=index1) # 将data的行列转换
data.iloc[0:4,0:2]
data.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_T.csv', header=0)
#处理
rpm_t = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_T.csv',engine='python')
rpm_t.info()
rpm_qn = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_qn.csv',engine='python')
rpm_qn.info()
rpm_qn.iloc[0:2,0:2]

for i in range(0,387):
    for j in range(0,452):
        if pd.isnull(rpm_t.iloc[i,j]):
            rpm_qn.iloc[i,j] = rpm_t.iloc[i,j]
        else:
            rpm_qn.iloc[i,j] = rpm_qn.iloc[i,j]
nan = rpm_qn.isnull().sum(axis=0)
rpm_qn.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_qn_NA.csv',index=0)
# 填充
rpm_qn_na = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_qn_NA.csv',engine='python')
rpm_qn_na.iloc[0:2,0:2]
#转置
data = rpm_qn_na.values  
index1 = list(rpm_qn_na.keys())  
data = list(map(list, zip(*data)))  
data = pd.DataFrame(data, index=index1)  
data.iloc[0:4,0:2]
data.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_qn_NA_T.csv', header=0)
#处理
t1 = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_qn_NA_T.csv', engine='python')
t1.iloc[0:3,0:3]
t2 = t1.drop('Gene',axis=1)
imputer = KNNImputer(n_neighbors=5,weights='distance') 
imputed = imputer.fit_transform(t2) 
df_imputed = pd.DataFrame(imputed, columns=t2.columns) 
df_imputed.iloc[0:2,0:2]
df_imputed.insert(0,'Gene',t1['Gene'])
type(df_imputed)
df_imputed.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_knn2.csv',index=0)

## log2
rpm_knn2 = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_knn2.csv',engine='python')
rpm_knn2_log2 = rpm_knn2.drop('Gene',axis=1)
rpm_knn2_log2 = np.log2(rpm_knn2_log2)
rpm_knn2_log2.insert(0,'Gene',rpm_knn2['Gene']) 
rpm_knn2_log2.iloc[0:2,0:2]
rpm_knn2_log2.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_log2.csv',index=0)

## z-score
from sklearn import preprocessing
rpm_knn2_log2 = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_log2.csv',engine='python')
rpm_knn2_log2.iloc[0:2,0:2]
rpm_knn2_log2 = rpm_knn2_log2.drop('Gene',axis=1)
values = rpm_knn2_log2.values 
values = values.astype('float32')  
data = preprocessing.scale(values) 
df = pd.DataFrame(data)  
df.columns=rpm_knn2_log2.columns  
df.iloc[0:2,0:2]
df.insert(0,'Gene',rpm_knn2['Gene']) 
df.to_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_z-score.csv',index=0) 

## knn2按列计算均值和方差
a = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_knn2.csv',engine='python')
a.info()
a.iloc[0:2,0:2]
mean = a.mean().sort_values(ascending=False)
std = a.std().sort_values(ascending=False)
skew = a.skew().sort_values(ascending=False)
kurt = a.kurt().sort_values(ascending=False)
r = pd.concat([mean,std,skew,kurt],axis=1) # 自动匹配
r['cov'] = r[1]/r[0]
r.to_csv('E:\\sirebrowser\\OV\\miRNA\\388miRNAs.csv',index=1)

import numpy as np
import pandas as pd
import keras
from keras.datasets import reuters
import tensorflow-gpu
from tensorflow.python.client import device_lib
import tensorflow as tf
print(device_lib.list_local_devices())
print(tf.test.is_built_with_cuda())

## 准备cox数据
z = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452samples+388miRNAs_z-score.csv',engine='python')
z.iloc[0:2,0:2]
z = z.set_index('Gene',drop=False)

c = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452clin_info.csv',engine='python')
c.info()
cc = c.iloc[:,[0,2,1]]
cc = cc.replace('alive',1)
cc = cc.replace('dead',2)
cc = cc.set_index('sample',drop=False)
cc = cc.iloc[:,[1,2]]
cox_data = pd.concat([z,cc],axis=1)
cox_data.to_csv('E:\\sirebrowser\\OV\\miRNA\\cox_data.csv',index=0)















