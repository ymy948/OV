# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:20:46 2021

@author: DELL
"""
import pandas as pd
import numpy as np
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.ensemble import  RandomForestClassifier
import matplotlib.pyplot as pltl
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from scipy import interp
########################################################### 数据准备1
data_1 = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\3\\17+18+21.csv', engine='python',encoding='UTF-8-sig')
data_1 = data_1.set_index('Sample')
data_1 = data_1.iloc[:,:20]
# data = data.drop('Sample',axis=1)
# 确保在每个group中分成0.7 0.3
g0_1 = data_1[data_1['kgroups'] == 0]
g1_1 = data_1[data_1['kgroups'] == 1]
g2_1 = data_1[data_1['kgroups'] == 2]
g3_1 = data_1[data_1['kgroups'] == 3]
g4_1 = data_1[data_1['kgroups'] == 4]
g5_1 = data_1[data_1['kgroups'] == 5]
g6_1 = data_1[data_1['kgroups'] == 6]

split = 19
x0_1,y0_1=np.split(g0_1,indices_or_sections=(split,),axis=1)
train_data0_1,test_data0_1,train_label0_1,test_label0_1 =train_test_split(x0_1,y0_1, random_state=1, train_size=0.7,test_size=0.3)
x1_1,y1_1=np.split(g1_1,indices_or_sections=(split,),axis=1) 
train_data1_1,test_data1_1,train_label1_1,test_label1_1 =train_test_split(x1_1,y1_1, random_state=1, train_size=0.7,test_size=0.3) 
x2_1,y2_1=np.split(g2_1,indices_or_sections=(split,),axis=1) 
train_data2_1,test_data2_1,train_label2_1,test_label2_1 =train_test_split(x2_1,y2_1, random_state=1, train_size=0.7,test_size=0.3) 
x3_1,y3_1=np.split(g3_1,indices_or_sections=(split,),axis=1)
train_data3_1,test_data3_1,train_label3_1,test_label3_1 =train_test_split(x3_1,y3_1, random_state=1, train_size=0.7,test_size=0.3)
x4_1,y4_1=np.split(g4_1,indices_or_sections=(split,),axis=1) 
train_data4_1,test_data4_1,train_label4_1,test_label4_1 =train_test_split(x4_1,y4_1, random_state=1, train_size=0.7,test_size=0.3) 
x5_1,y5_1=np.split(g5_1,indices_or_sections=(split,),axis=1) 
train_data5_1,test_data5_1,train_label5_1,test_label5_1 =train_test_split(x5_1,y5_1, random_state=1, train_size=0.7,test_size=0.3) 
x6_1,y6_1=np.split(g6_1,indices_or_sections=(split,),axis=1) 
train_data6_1,test_data6_1,train_label6_1,test_label6_1 =train_test_split(x6_1,y6_1, random_state=1, train_size=0.7,test_size=0.3) 

frames = [train_data0_1,train_data1_1,train_data2_1,train_data3_1,train_data4_1,train_data5_1,train_data6_1]
train_data_1 = pd.concat(frames)
frames = [test_data0_1,test_data1_1,test_data2_1,test_data3_1,test_data4_1,test_data5_1,test_data6_1]
test_data_1 = pd.concat(frames)
frames = [train_label0_1,train_label1_1,train_label2_1,train_label3_1,train_label4_1,train_label5_1,train_label6_1]
train_label_1 = pd.concat(frames)
frames = [test_label0_1,test_label1_1,test_label2_1,test_label3_1,test_label4_1,test_label5_1,test_label6_1]
test_label_1 = pd.concat(frames)
frames = [x0_1,x1_1,x2_1,x3_1,x4_1,x5_1,x6_1]
x_1 = pd.concat(frames)
frames = [y0_1,y1_1,y2_1,y3_1,y4_1,y5_1,y6_1]
y_1 = pd.concat(frames)

import sklearn.neural_network as snn
classifier_1 = snn.MLPClassifier(hidden_layer_sizes=(500,400),activation='relu',solver='sgd',alpha=0.0001,
                                 learning_rate='adaptive',learning_rate_init=0.001,
                                 max_iter=500,random_state=24)
classifier_1.fit(train_data_1, train_label_1.values.ravel())

print("训练集：",classifier_1.score(train_data_1,train_label_1))
print("测试集：",classifier_1.score(test_data_1,test_label_1))
########################################################### 数据准备2
data_2 = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\4\\17+18+19+21.csv', engine='python',encoding='UTF-8-sig')
data_2= data_2.set_index('Sample')
data_2 = data_2.iloc[:,:21]
# data = data.drop('Sample',axis=1)
# 确保在每个group中分成0.7 0.3
g0_2 = data_2[data_2['kgroups'] == 0]
g1_2 = data_2[data_2['kgroups'] == 1]
g2_2 = data_2[data_2['kgroups'] == 2]
g3_2 = data_2[data_2['kgroups'] == 3]
g4_2 = data_2[data_2['kgroups'] == 4]
g5_2 = data_2[data_2['kgroups'] == 5]
g6_2 = data_2[data_2['kgroups'] == 6]

split = 20
x0_2,y0_2=np.split(g0_2,indices_or_sections=(split,),axis=1)
train_data0_2,test_data0_2,train_label0_2,test_label0_2 =train_test_split(x0_2,y0_2, random_state=1, train_size=0.7,test_size=0.3)
x1_2,y1_2=np.split(g1_2,indices_or_sections=(split,),axis=1) 
train_data1_2,test_data1_2,train_label1_2,test_label1_2 =train_test_split(x1_2,y1_2, random_state=1, train_size=0.7,test_size=0.3) 
x2_2,y2_2=np.split(g2_2,indices_or_sections=(split,),axis=1) 
train_data2_2,test_data2_2,train_label2_2,test_label2_2 =train_test_split(x2_2,y2_2, random_state=1, train_size=0.7,test_size=0.3) 
x3_2,y3_2=np.split(g3_2,indices_or_sections=(split,),axis=1)
train_data3_2,test_data3_2,train_label3_2,test_label3_2 =train_test_split(x3_2,y3_2, random_state=1, train_size=0.7,test_size=0.3)
x4_2,y4_2=np.split(g4_2,indices_or_sections=(split,),axis=1) 
train_data4_2,test_data4_2,train_label4_2,test_label4_2 =train_test_split(x4_2,y4_2, random_state=1, train_size=0.7,test_size=0.3) 
x5_2,y5_2=np.split(g5_2,indices_or_sections=(split,),axis=1) 
train_data5_2,test_data5_2,train_label5_2,test_label5_2 =train_test_split(x5_2,y5_2, random_state=1, train_size=0.7,test_size=0.3) 
x6_2,y6_2=np.split(g6_2,indices_or_sections=(split,),axis=1) 
train_data6_2,test_data6_2,train_label6_2,test_label6_2 =train_test_split(x6_2,y6_2, random_state=1, train_size=0.7,test_size=0.3) 

frames = [train_data0_2,train_data1_2,train_data2_2,train_data3_2,train_data4_2,train_data5_2,train_data6_2]
train_data_2 = pd.concat(frames)
frames = [test_data0_2,test_data1_2,test_data2_2,test_data3_2,test_data4_2,test_data5_2,test_data6_2]
test_data_2 = pd.concat(frames)
frames = [train_label0_2,train_label1_2,train_label2_2,train_label3_2,train_label4_2,train_label5_2,train_label6_2]
train_label_2 = pd.concat(frames)
frames = [test_label0_2,test_label1_2,test_label2_2,test_label3_2,test_label4_2,test_label5_2,test_label6_2]
test_label_2 = pd.concat(frames)
frames = [x0_2,x1_2,x2_2,x3_2,x4_2,x5_2,x6_2]
x_2 = pd.concat(frames)
frames = [y0_2,y1_2,y2_2,y3_2,y4_2,y5_2,y6_2]
y_2 = pd.concat(frames)

import sklearn.neural_network as snn
classifier_2 = snn.MLPClassifier(hidden_layer_sizes=(500,400),activation='relu',solver='sgd',alpha=0.0001,
                                 learning_rate='adaptive',learning_rate_init=0.001,
                                 max_iter=500,random_state=24)
classifier_2.fit(train_data_2, train_label_2.values.ravel())

print("训练集：",classifier_2.score(train_data_2,train_label_2))
print("测试集：",classifier_2.score(test_data_2,test_label_2))
########################################################### 数据准备3
data_3 = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\km\\22k=7.csv', engine='python',encoding='UTF-8-sig')
data_3 = data_3.set_index('Sample')
data_3 = data_3.iloc[:,:23]
# data = data.drop('Sample',axis=1)
# 确保在每个group中分成0.7 0.3
g0_3 = data_3[data_3['kgroups'] == 0]
g1_3 = data_3[data_3['kgroups'] == 1]
g2_3 = data_3[data_3['kgroups'] == 2]
g3_3 = data_3[data_3['kgroups'] == 3]
g4_3 = data_3[data_3['kgroups'] == 4]
g5_3 = data_3[data_3['kgroups'] == 5]
g6_3 = data_3[data_3['kgroups'] == 6]

split = 22
x0_3,y0_3=np.split(g0_3,indices_or_sections=(split,),axis=1)
train_data0_3,test_data0_3,train_label0_3,test_label0_3 =train_test_split(x0_3,y0_3, random_state=1, train_size=0.7,test_size=0.3)
x1_3,y1_3=np.split(g1_3,indices_or_sections=(split,),axis=1) 
train_data1_3,test_data1_3,train_label1_3,test_label1_3 =train_test_split(x1_3,y1_3, random_state=1, train_size=0.7,test_size=0.3) 
x2_3,y2_3=np.split(g2_3,indices_or_sections=(split,),axis=1) 
train_data2_3,test_data2_3,train_label2_3,test_label2_3 =train_test_split(x2_3,y2_3, random_state=1, train_size=0.7,test_size=0.3) 
x3_3,y3_3=np.split(g3_3,indices_or_sections=(split,),axis=1)
train_data3_3,test_data3_3,train_label3_3,test_label3_3 =train_test_split(x3_3,y3_3, random_state=1, train_size=0.7,test_size=0.3)
x4_3,y4_3=np.split(g4_3,indices_or_sections=(split,),axis=1) 
train_data4_3,test_data4_3,train_label4_3,test_label4_3 =train_test_split(x4_3,y4_3, random_state=1, train_size=0.7,test_size=0.3) 
x5_3,y5_3=np.split(g5_3,indices_or_sections=(split,),axis=1) 
train_data5_3,test_data5_3,train_label5_3,test_label5_3 =train_test_split(x5_3,y5_3, random_state=1, train_size=0.7,test_size=0.3) 
x6_3,y6_3=np.split(g6_3,indices_or_sections=(split,),axis=1) 
train_data6_3,test_data6_3,train_label6_3,test_label6_3 =train_test_split(x6_3,y6_3, random_state=1, train_size=0.7,test_size=0.3) 

frames = [train_data0_3,train_data1_3,train_data2_3,train_data3_3,train_data4_3,train_data5_3,train_data6_3]
train_data_3 = pd.concat(frames)
frames = [test_data0_3,test_data1_3,test_data2_3,test_data3_3,test_data4_3,test_data5_3,test_data6_3]
test_data_3 = pd.concat(frames)
frames = [train_label0_3,train_label1_3,train_label2_3,train_label3_3,train_label4_3,train_label5_3,train_label6_3]
train_label_3 = pd.concat(frames)
frames = [test_label0_3,test_label1_3,test_label2_3,test_label3_3,test_label4_3,test_label5_3,test_label6_3]
test_label_3 = pd.concat(frames)
frames = [x0_3,x1_3,x2_3,x3_3,x4_3,x5_3,x6_3]
x_3 = pd.concat(frames)
frames = [y0_3,y1_3,y2_3,y3_3,y4_3,y5_3,y6_3]
y_3 = pd.concat(frames)

import sklearn.neural_network as snn
classifier_3 = snn.MLPClassifier(hidden_layer_sizes=(500,400),activation='relu',solver='sgd',alpha=0.0001,
                                 learning_rate='adaptive',learning_rate_init=0.001,
                                 max_iter=500,random_state=24)
classifier_3.fit(train_data_3, train_label_3.values.ravel())

print("训练集：",classifier_3.score(train_data_3,train_label_3))
print("测试集：",classifier_3.score(test_data_3,test_label_3))

############################################# roc画图
fontt={'color': 'k',
      'size': 50,
      'family': 'Arial'}
fonty={'color': 'k',
      'size': 20,
      'family': 'Arial'}
font={'color': 'k',
      'size': 10,
      'family': 'Arial'}

x_nd_1 = x_1.values
y_nd_1 = y_1.values
y_1 = label_binarize(y_nd_1, classes=[0,1,2,3,4,5,6])
n_classes_1 = y_1.shape[1]
n_samples_1, n_features_1 = x_nd_1.shape
train_label_1_1 = label_binarize(train_label_1, classes=[0,1,2,3,4,5,6])
test_label_1_1 = label_binarize(test_label_1, classes=[0,1,2,3,4,5,6])

x_nd_2 = x_2.values
y_nd_2 = y_2.values
y_2 = label_binarize(y_nd_2, classes=[0,1,2,3,4,5,6])
n_classes_2 = y_2.shape[1]
n_samples_2, n_features_2 = x_nd_2.shape
train_label_1_2 = label_binarize(train_label_2, classes=[0,1,2,3,4,5,6])
test_label_1_2 = label_binarize(test_label_2, classes=[0,1,2,3,4,5,6])

x_nd_3 = x_3.values
y_nd_3 = y_3.values
y_3 = label_binarize(y_nd_3, classes=[0,1,2,3,4,5,6])
n_classes_3 = y_3.shape[1]
n_samples_3, n_features_3 = x_nd_3.shape
train_label_1_3 = label_binarize(train_label_3, classes=[0,1,2,3,4,5,6])
test_label_1_3 = label_binarize(test_label_3, classes=[0,1,2,3,4,5,6])

##############################################################test
#train_data = train_data.drop('Sample',axis=1)
#test_data = test_data.drop('Sample',axis=1)
##############################################
y_score_1 = classifier_1.fit(train_data_1,train_label_1.values.ravel()).predict_proba(test_data_1)

fpr_1 = dict()
tpr_1 = dict()
roc_auc_1 = dict()
for i in range(n_classes_1):
    fpr_1[i], tpr_1[i], _ = roc_curve(test_label_1_1[:, i], y_score_1[:, i])
    roc_auc_1[i] = auc(fpr_1[i], tpr_1[i])
    
fpr_1["micro"], tpr_1["micro"], _ = roc_curve(test_label_1_1.ravel(), y_score_1.ravel())
roc_auc_1["micro"] = auc(fpr_1["micro"], tpr_1["micro"])
all_fpr_1 = np.unique(np.concatenate([fpr_1[i] for i in range(n_classes_1)]))
mean_tpr_1 = np.zeros_like(all_fpr_1)
for i in range(n_classes_1):
    mean_tpr_1 += interp(all_fpr_1, fpr_1[i], tpr_1[i])
# Finally average it and compute AUC
mean_tpr_1 /= n_classes_1
fpr_1["macro"] = all_fpr_1
tpr_1["macro"] = mean_tpr_1
roc_auc_1["macro"] = auc(fpr_1["macro"], tpr_1["macro"])
##############################################
y_score_2 = classifier_2.fit(train_data_2,train_label_2.values.ravel()).predict_proba(test_data_2)

fpr_2 = dict()
tpr_2 = dict()
roc_auc_2 = dict()
for i in range(n_classes_2):
    fpr_2[i], tpr_2[i], _ = roc_curve(test_label_1_2[:, i], y_score_2[:, i])
    roc_auc_2[i] = auc(fpr_2[i], tpr_2[i])
    
fpr_2["micro"], tpr_2["micro"], _ = roc_curve(test_label_1_2.ravel(), y_score_2.ravel())
roc_auc_2["micro"] = auc(fpr_2["micro"], tpr_2["micro"])
all_fpr_2 = np.unique(np.concatenate([fpr_2[i] for i in range(n_classes_2)]))
mean_tpr_2 = np.zeros_like(all_fpr_2)
for i in range(n_classes_2):
    mean_tpr_2 += interp(all_fpr_2, fpr_2[i], tpr_2[i])
# Finally average it and compute AUC
mean_tpr_2 /= n_classes_2
fpr_2["macro"] = all_fpr_2
tpr_2["macro"] = mean_tpr_2
roc_auc_2["macro"] = auc(fpr_2["macro"], tpr_2["macro"])
##############################################
y_score_3 = classifier_3.fit(train_data_3,train_label_3.values.ravel()).predict_proba(test_data_3)

fpr_3 = dict()
tpr_3 = dict()
roc_auc_3 = dict()
for i in range(n_classes_3):
    fpr_3[i], tpr_3[i], _ = roc_curve(test_label_1_3[:, i], y_score_3[:, i])
    roc_auc_3[i] = auc(fpr_3[i], tpr_3[i])
    
fpr_3["micro"], tpr_3["micro"], _ = roc_curve(test_label_1_3.ravel(), y_score_3.ravel())
roc_auc_3["micro"] = auc(fpr_3["micro"], tpr_3["micro"])
all_fpr_3 = np.unique(np.concatenate([fpr_3[i] for i in range(n_classes_3)]))
mean_tpr_3 = np.zeros_like(all_fpr_3)
for i in range(n_classes_3):
    mean_tpr_3 += interp(all_fpr_3, fpr_3[i], tpr_3[i])
# Finally average it and compute AUC
mean_tpr_3 /= n_classes_3
fpr_3["macro"] = all_fpr_3
tpr_3["macro"] = mean_tpr_3
roc_auc_3["macro"] = auc(fpr_3["macro"], tpr_3["macro"])

#macro
lw = 2   #'               {0:0.4f}'
plt.figure(figsize=(8,8))
plt.plot(fpr_1["macro"], tpr_1["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_1["macro"]),
         color='navy', linestyle='-', linewidth=8)

plt.plot(fpr_2["macro"], tpr_2["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_2["macro"]),
         color='#7E6148FF', linestyle='-', linewidth=8)

plt.plot(fpr_3["macro"], tpr_3["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_3["macro"]),
         color='#F64B35FF', linestyle='-', linewidth=8)
 
plt.plot([0, 1], [0, 1], 'k--', lw=5)
plt.xlim([-0.02, 1.3])
plt.ylim([-0.02, 1.05])
plt.legend(prop={'family' : 'Arial', 'size'   : 38},handletextpad=0.5,frameon=False,labelspacing=0.1,loc=(0.2,-0.03))
plt.tick_params(width=6)
plt.tick_params(length=6)

ax=plt.gca()
ax.spines['bottom'].set_linewidth('4')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('4')
ax.spines['right'].set_linewidth('0')
plt.title('ROC of test set',fontdict=fontt)
# plt.xlabel('False Positive Rate',fontdict=fonty)
# plt.ylabel('True Positive Rate',fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 40)
plt.xticks(fontproperties = 'Arial', size = 40,rotation=45)
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\test_roc.png',dpi=1500, bbox_inches = 'tight')
plt.close('all')

#########################################################################train
##############################################
y_score_train_1 = classifier_1.fit(train_data_1,train_label_1.values.ravel()).predict_proba(train_data_1)

fpr_train_1 = dict()
tpr_train_1 = dict()
roc_auc_train_1 = dict()
for i in range(n_classes_1):
    fpr_train_1[i], tpr_train_1[i], _ = roc_curve(train_label_1_1[:, i], y_score_train_1[:, i])
    roc_auc_train_1[i] = auc(fpr_train_1[i], tpr_train_1[i])
    
fpr_train_1["micro"], tpr_train_1["micro"], _ = roc_curve(train_label_1_1.ravel(), y_score_train_1.ravel())
roc_auc_train_1["micro"] = auc(fpr_train_1["micro"], tpr_train_1["micro"])
all_fpr_train_1 = np.unique(np.concatenate([fpr_train_1[i] for i in range(n_classes_1)]))
mean_tpr_train_1 = np.zeros_like(all_fpr_train_1)
for i in range(n_classes_1):
    mean_tpr_train_1 += interp(all_fpr_train_1, fpr_train_1[i], tpr_train_1[i])
# Finally average it and compute AUC
mean_tpr_train_1 /= n_classes_1
fpr_train_1["macro"] = all_fpr_train_1
tpr_train_1["macro"] = mean_tpr_train_1
roc_auc_train_1["macro"] = auc(fpr_train_1["macro"], tpr_train_1["macro"])
##############################################
y_score_train_2 = classifier_2.fit(train_data_2,train_label_2.values.ravel()).predict_proba(train_data_2)

fpr_train_2 = dict()
tpr_train_2 = dict()
roc_auc_train_2 = dict()
for i in range(n_classes_2):
    fpr_train_2[i], tpr_train_2[i], _ = roc_curve(train_label_1_2[:, i], y_score_train_2[:, i])
    roc_auc_train_2[i] = auc(fpr_train_2[i], tpr_train_2[i])
    
fpr_train_2["micro"], tpr_train_2["micro"], _ = roc_curve(train_label_1_2.ravel(), y_score_train_2.ravel())
roc_auc_train_2["micro"] = auc(fpr_train_2["micro"], tpr_train_2["micro"])
all_fpr_train_2 = np.unique(np.concatenate([fpr_train_2[i] for i in range(n_classes_2)]))
mean_tpr_train_2 = np.zeros_like(all_fpr_train_2)
for i in range(n_classes_2):
    mean_tpr_train_2 += interp(all_fpr_train_2, fpr_train_2[i], tpr_train_2[i])
# Finally average it and compute AUC
mean_tpr_train_2 /= n_classes_2
fpr_train_2["macro"] = all_fpr_train_2
tpr_train_2["macro"] = mean_tpr_train_2
roc_auc_train_2["macro"] = auc(fpr_train_2["macro"], tpr_train_2["macro"])
##############################################
y_score_train_3 = classifier_3.fit(train_data_3,train_label_3.values.ravel()).predict_proba(train_data_3)

fpr_train_3 = dict()
tpr_train_3 = dict()
roc_auc_train_3 = dict()
for i in range(n_classes_3):
    fpr_train_3[i], tpr_train_3[i], _ = roc_curve(train_label_1_3[:, i], y_score_train_3[:, i])
    roc_auc_train_3[i] = auc(fpr_train_3[i], tpr_train_3[i])
    
fpr_train_3["micro"], tpr_train_3["micro"], _ = roc_curve(train_label_1_3.ravel(), y_score_train_3.ravel())
roc_auc_train_3["micro"] = auc(fpr_train_3["micro"], tpr_train_3["micro"])
all_fpr_train_3 = np.unique(np.concatenate([fpr_train_3[i] for i in range(n_classes_3)]))
mean_tpr_train_3 = np.zeros_like(all_fpr_train_3)
for i in range(n_classes_3):
    mean_tpr_train_3 += interp(all_fpr_train_3, fpr_train_3[i], tpr_train_3[i])
# Finally average it and compute AUC
mean_tpr_train_3 /= n_classes_3
fpr_train_3["macro"] = all_fpr_train_3
tpr_train_3["macro"] = mean_tpr_train_3
roc_auc_train_3["macro"] = auc(fpr_train_3["macro"], tpr_train_3["macro"])

#########################3macro
lw = 2
plt.figure(figsize=(8,8))
plt.plot(fpr_train_1["macro"], tpr_train_1["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_train_1["macro"]),
         color='navy', linestyle='-', linewidth=8)

plt.plot(fpr_train_2["macro"], tpr_train_2["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_train_2["macro"]),
         color='#7E6148FF', linestyle='-', linewidth=8)

plt.plot(fpr_train_3["macro"], tpr_train_3["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_train_3["macro"]),
         color='#F64B35FF', linestyle='-', linewidth=8)
 
plt.plot([0, 1], [0, 1], 'k--', lw=5)
plt.xlim([-0.02, 1.3])
plt.ylim([-0.02, 1.05])
plt.legend(prop={'family' : 'Arial', 'size'   : 38},handletextpad=0.5,frameon=False,labelspacing=0.1,loc=(0.2,-0.03))
plt.tick_params(width=6)
plt.tick_params(length=6)

ax=plt.gca()
ax.spines['bottom'].set_linewidth('4')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('4')
ax.spines['right'].set_linewidth('0')
plt.title('ROC of train set',fontdict=fontt)
# plt.xlabel('False Positive Rate',fontdict=fonty)
# plt.ylabel('True Positive Rate',fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 40)
plt.xticks(fontproperties = 'Arial', size = 40,rotation=45)
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\train_roc.png',dpi=1500, bbox_inches = 'tight')
plt.close('all')
#######################################################################combine
##############################################
data_1 = pd.concat([train_data_1,test_data_1],axis=0)
label_1 = pd.concat([train_label_1,test_label_1],axis=0)
label_1 = label_binarize(label_1, classes=[0,1,2,3,4,5,6])

y_score_c_1 = classifier_1.fit(train_data_1,train_label_1.values.ravel()).predict_proba(data_1)

fpr_c_1 = dict()
tpr_c_1 = dict()
roc_auc_c_1 = dict()
for i in range(n_classes_1):
    fpr_c_1[i], tpr_c_1[i], _ = roc_curve(label_1[:, i], y_score_c_1[:, i])
    roc_auc_c_1[i] = auc(fpr_c_1[i], tpr_c_1[i])
    
fpr_c_1["micro"], tpr_c_1["micro"], _ = roc_curve(label_1.ravel(), y_score_c_1.ravel())
roc_auc_c_1["micro"] = auc(fpr_c_1["micro"], tpr_c_1["micro"])
all_fpr_c_1 = np.unique(np.concatenate([fpr_c_1[i] for i in range(n_classes_1)]))
mean_tpr_c_1 = np.zeros_like(all_fpr_c_1)
for i in range(n_classes_1):
    mean_tpr_c_1 += interp(all_fpr_c_1, fpr_c_1[i], tpr_c_1[i])
# Finally average it and compute AUC
mean_tpr_c_1 /= n_classes_1
fpr_c_1["macro"] = all_fpr_c_1
tpr_c_1["macro"] = mean_tpr_c_1
roc_auc_c_1["macro"] = auc(fpr_c_1["macro"], tpr_c_1["macro"])
##############################################
data_2 = pd.concat([train_data_2,test_data_2],axis=0)
label_2 = pd.concat([train_label_2,test_label_2],axis=0)
label_2 = label_binarize(label_2, classes=[0,1,2,3,4,5,6])

y_score_c_2 = classifier_2.fit(train_data_2,train_label_2.values.ravel()).predict_proba(data_2)

fpr_c_2 = dict()
tpr_c_2 = dict()
roc_auc_c_2 = dict()
for i in range(n_classes_2):
    fpr_c_2[i], tpr_c_2[i], _ = roc_curve(label_2[:, i], y_score_c_2[:, i])
    roc_auc_c_2[i] = auc(fpr_c_2[i], tpr_c_2[i])
    
fpr_c_2["micro"], tpr_c_2["micro"], _ = roc_curve(label_2.ravel(), y_score_c_2.ravel())
roc_auc_c_2["micro"] = auc(fpr_c_2["micro"], tpr_c_2["micro"])
all_fpr_c_2 = np.unique(np.concatenate([fpr_c_2[i] for i in range(n_classes_2)]))
mean_tpr_c_2 = np.zeros_like(all_fpr_c_2)
for i in range(n_classes_2):
    mean_tpr_c_2 += interp(all_fpr_c_2, fpr_c_2[i], tpr_c_2[i])
# Finally average it and compute AUC
mean_tpr_c_2 /= n_classes_2
fpr_c_2["macro"] = all_fpr_c_2
tpr_c_2["macro"] = mean_tpr_c_2
roc_auc_c_2["macro"] = auc(fpr_c_2["macro"], tpr_c_2["macro"])
##############################################
data_3 = pd.concat([train_data_3,test_data_3],axis=0)
label_3 = pd.concat([train_label_3,test_label_3],axis=0)
label_3 = label_binarize(label_3, classes=[0,1,2,3,4,5,6])

y_score_c_3 = classifier_3.fit(train_data_3,train_label_3.values.ravel()).predict_proba(data_3)

fpr_c_3 = dict()
tpr_c_3 = dict()
roc_auc_c_3 = dict()
for i in range(n_classes_3):
    fpr_c_3[i], tpr_c_3[i], _ = roc_curve(label_3[:, i], y_score_c_3[:, i])
    roc_auc_c_3[i] = auc(fpr_c_3[i], tpr_c_3[i])
    
fpr_c_3["micro"], tpr_c_3["micro"], _ = roc_curve(label_3.ravel(), y_score_c_3.ravel())
roc_auc_c_3["micro"] = auc(fpr_c_3["micro"], tpr_c_3["micro"])
all_fpr_c_3 = np.unique(np.concatenate([fpr_c_3[i] for i in range(n_classes_3)]))
mean_tpr_c_3 = np.zeros_like(all_fpr_c_3)
for i in range(n_classes_3):
    mean_tpr_c_3 += interp(all_fpr_c_3, fpr_c_3[i], tpr_c_3[i])
# Finally average it and compute AUC
mean_tpr_c_3 /= n_classes_3
fpr_c_3["macro"] = all_fpr_c_3
tpr_c_3["macro"] = mean_tpr_c_3
roc_auc_c_3["macro"] = auc(fpr_c_3["macro"], tpr_c_3["macro"])

#macro
lw = 2
plt.figure(figsize=(8,8))
plt.plot(fpr_c_1["macro"], tpr_c_1["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_c_1["macro"]),
         color='navy', linestyle='-', linewidth=8)

plt.plot(fpr_c_2["macro"], tpr_c_2["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_c_2["macro"]),
         color='#7E6148FF', linestyle='-', linewidth=8)

plt.plot(fpr_c_3["macro"], tpr_c_3["macro"],
         label='AUC: {0:0.4f}'
               ''.format(roc_auc_c_3["macro"]),
         color='#F64B35FF', linestyle='-', linewidth=8)
 
plt.plot([0, 1], [0, 1], 'k--', lw=5)
plt.xlim([-0.02, 1.3])
plt.ylim([-0.02, 1.05])
plt.legend(prop={'family':'Arial','size': 38},handletextpad=0.5,frameon=False,labelspacing=0.1,loc=(0.2,-0.03))
plt.tick_params(width=6)
plt.tick_params(length=6)

ax=plt.gca()
ax.spines['bottom'].set_linewidth('4')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('4')
ax.spines['right'].set_linewidth('0')
plt.title('ROC of all set',fontdict=fontt)
# plt.xlabel('False Positive Rate',fontdict=fonty)
# plt.ylabel('True Positive Rate',fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 40)
plt.xticks(fontproperties = 'Arial', size = 40,rotation=45)
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\all_roc.png',dpi=1500, bbox_inches = 'tight')
plt.close('all')