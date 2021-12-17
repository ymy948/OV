# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:23:18 2021

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

data = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\4\\17+18+19+21.csv', engine='python')

data = data.set_index('Sample')
# data = data.drop('Sample',axis=1)
# 确保在每个group中分成0.7 0.3
g0 = data[data['kgroups'] == 0]
g0.info()
g1 = data[data['kgroups'] == 1]
g1.info()
g2 = data[data['kgroups'] == 2]
g2.info()
g3 = data[data['kgroups'] == 3]
g3.info()
g4 = data[data['kgroups'] == 4]
g4.info()
g5 = data[data['kgroups'] == 5]
g5.info()
g6 = data[data['kgroups'] == 6]
g6.info()

x0,y0=np.split(g0,indices_or_sections=(20,),axis=1)
train_data0,test_data0,train_label0,test_label0 =train_test_split(x0,y0, random_state=1, train_size=0.7,test_size=0.3)
x1,y1=np.split(g1,indices_or_sections=(20,),axis=1) 
train_data1,test_data1,train_label1,test_label1 =train_test_split(x1,y1, random_state=1, train_size=0.7,test_size=0.3) 
x2,y2=np.split(g2,indices_or_sections=(20,),axis=1) 
train_data2,test_data2,train_label2,test_label2 =train_test_split(x2,y2, random_state=1, train_size=0.7,test_size=0.3) 
x3,y3=np.split(g3,indices_or_sections=(20,),axis=1)
train_data3,test_data3,train_label3,test_label3 =train_test_split(x3,y3, random_state=1, train_size=0.7,test_size=0.3)
x4,y4=np.split(g4,indices_or_sections=(20,),axis=1) 
train_data4,test_data4,train_label4,test_label4 =train_test_split(x4,y4, random_state=1, train_size=0.7,test_size=0.3) 
x5,y5=np.split(g5,indices_or_sections=(20,),axis=1) 
train_data5,test_data5,train_label5,test_label5 =train_test_split(x5,y5, random_state=1, train_size=0.7,test_size=0.3) 
x6,y6=np.split(g6,indices_or_sections=(20,),axis=1) 
train_data6,test_data6,train_label6,test_label6 =train_test_split(x6,y6, random_state=1, train_size=0.7,test_size=0.3) 

frames = [train_data0,train_data1,train_data2,train_data3,train_data4,train_data5,train_data6]
train_data = pd.concat(frames)
train_data.info()
frames = [test_data0,test_data1,test_data2,test_data3,test_data4,test_data5,test_data6]
test_data = pd.concat(frames)
test_data.info()
frames = [train_label0,train_label1,train_label2,train_label3,train_label4,train_label5,train_label6]
train_label = pd.concat(frames)
train_label.info()
frames = [test_label0,test_label1,test_label2,test_label3,test_label4,test_label5,test_label6]
test_label = pd.concat(frames)
test_label.info()
frames = [x0,x1,x2,x3,x4,x5,x6]
x = pd.concat(frames)
frames = [y0,y1,y2,y3,y4,y5,y6]
y = pd.concat(frames)

################# random forest选择23个中重要的
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.ensemble import  RandomForestClassifier
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from scipy import interp
x_nd = x.values
y_nd = y.values
#y = label_binarize(y_nd, classes=[0,1,2,3,4,5,6])
n_classes = y.shape[1]
n_samples, n_features = x_nd.shape

train_data_1 = train_data.values
train_label_1 = label_binarize(train_label, classes=[0,1,2,3,4,5,6])
test_label_1 = label_binarize(test_label, classes=[0,1,2,3,4,5,6])


clf=RandomForestClassifier(random_state = 1, class_weight="balanced")
g = GridSearchCV(estimator=clf,cv=41, n_jobs=6,scoring = 'accuracy',param_grid={'n_estimators':range(10,300,5)})
g.fit(x,y)
print(g.best_params_)
print(g.best_score_) #cv=10-195 cv=41-155

# random forest 
clf=RandomForestClassifier(n_estimators =155, random_state = 1, class_weight="balanced")
output = cross_validate(clf, x_nd, y_nd, cv=10, scoring = 'accuracy', return_estimator =True,n_jobs=4)

for idx,estimator in enumerate(output['estimator']):
     feature_importances = pd.DataFrame(estimator.feature_importances_,
                                        index = train_data.columns,
                                         columns=['importance']).sort_values('importance', ascending=False)
     print(feature_importances)

s = feature_importances[0]

for idx,estimator in enumerate(output['estimator']):
    feature_importances[idx] = pd.DataFrame(estimator.feature_importances_,
                                       index = train_data.columns,
                                        columns=['importance'])
    print(feature_importances[idx])
##     s[idx] = feature_importances  
    s = pd.concat([s,feature_importances[idx]],axis=1)
    
s.head()
s = s.drop('0',axis=1)
s = s.iloc[:,2:]

sm41 = s.mean(1).sort_values(ascending=False)
sm10 = s.mean(1).sort_values(ascending=False)
sm = pd.concat([sm10,sm41],axis=1)
sm.insert(0,'miRNA',sm.index)

sm.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\rf\\randomforest1.csv',index=0) #保存结果

############################################# svm
data = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\rf\\22k=7rf.csv', engine='python')
data = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,24]] # top15
data = data.set_index('Sample')
# data = data.drop('Sample',axis=1)
# 确保在每个group中分成0.7 0.3
g0 = data[data['kgroups'] == 0]
g0.info()
g1 = data[data['kgroups'] == 1]
g1.info()
g2 = data[data['kgroups'] == 2]
g2.info()
g3 = data[data['kgroups'] == 3]
g3.info()
g4 = data[data['kgroups'] == 4]
g4.info()
g5 = data[data['kgroups'] == 5]
g5.info()
g6 = data[data['kgroups'] == 6]
g6.info()

x0,y0=np.split(g0,indices_or_sections=(19,),axis=1)
train_data0,test_data0,train_label0,test_label0 =train_test_split(x0,y0, random_state=2, train_size=0.7,test_size=0.3)
x1,y1=np.split(g1,indices_or_sections=(19,),axis=1) 
train_data1,test_data1,train_label1,test_label1 =train_test_split(x1,y1, random_state=2, train_size=0.7,test_size=0.3) 
x2,y2=np.split(g2,indices_or_sections=(19,),axis=1) 
train_data2,test_data2,train_label2,test_label2 =train_test_split(x2,y2, random_state=2, train_size=0.7,test_size=0.3) 
x3,y3=np.split(g3,indices_or_sections=(19,),axis=1)
train_data3,test_data3,train_label3,test_label3 =train_test_split(x3,y3, random_state=2, train_size=0.7,test_size=0.3)
x4,y4=np.split(g4,indices_or_sections=(19,),axis=1) 
train_data4,test_data4,train_label4,test_label4 =train_test_split(x4,y4, random_state=2, train_size=0.7,test_size=0.3) 
x5,y5=np.split(g5,indices_or_sections=(19,),axis=1) 
train_data5,test_data5,train_label5,test_label5 =train_test_split(x5,y5, random_state=2, train_size=0.7,test_size=0.3) 
x6,y6=np.split(g6,indices_or_sections=(19,),axis=1) 
train_data6,test_data6,train_label6,test_label6 =train_test_split(x6,y6, random_state=2, train_size=0.7,test_size=0.3) 

frames = [train_data0,train_data1,train_data2,train_data3,train_data4,train_data5,train_data6]
train_data = pd.concat(frames)
train_data.info()
frames = [test_data0,test_data1,test_data2,test_data3,test_data4,test_data5,test_data6]
test_data = pd.concat(frames)
test_data.info()
frames = [train_label0,train_label1,train_label2,train_label3,train_label4,train_label5,train_label6]
train_label = pd.concat(frames)
train_label.info()
frames = [test_label0,test_label1,test_label2,test_label3,test_label4,test_label5,test_label6]
test_label = pd.concat(frames)
test_label.info()
frames = [x0,x1,x2,x3,x4,x5,x6]
x = pd.concat(frames)
frames = [y0,y1,y2,y3,y4,y5,y6]
y = pd.concat(frames)

# 直接训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=0.05,decision_function_shape='ovo', probability=True)  # ovr:一对多策略
classifier.fit(train_data, train_label.values.ravel())  # ravel函数在降维时默认是行序优先
#交叉验证
scores = cross_val_score(classifier, x, y, cv=10)
scores.mean() #0.8695
print("训练集：",classifier.score(train_data,train_label)) #1.0
print("测试集：",classifier.score(test_data,test_label)) #0.8273

## 10折训练svm分类器(在train_data中)
train_set0 = pd.concat([train_data0,train_label0],axis=1)
train_set1 = pd.concat([train_data1,train_label1],axis=1)
train_set2 = pd.concat([train_data2,train_label2],axis=1)
train_set3 = pd.concat([train_data3,train_label3],axis=1)
train_set4 = pd.concat([train_data4,train_label4],axis=1)
train_set5 = pd.concat([train_data5,train_label5],axis=1)
train_set6 = pd.concat([train_data6,train_label6],axis=1)

# 不放回抽样  
#0
train_set00 = train_set0.sample(n=5,replace=False)
for i in train_set00.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set01 = train_set0.sample(n=5,replace=False)
for i in train_set01.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set02 = train_set0.sample(n=5,replace=False)
for i in train_set02.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set03 = train_set0.sample(n=5,replace=False)
for i in train_set03.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set04 = train_set0.sample(n=6,replace=False)
for i in train_set04.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set05 = train_set0.sample(n=6,replace=False)
for i in train_set05.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set06 = train_set0.sample(n=6,replace=False)
for i in train_set06.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set07 = train_set0.sample(n=6,replace=False)
for i in train_set07.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set08 = train_set0.sample(n=6,replace=False)
for i in train_set08.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
train_set09 = train_set0.sample(n=6,replace=False)
for i in train_set09.index:
    print(i)
    train_set0 = train_set0.drop(i,axis=0)
    
#1
train_set10 = train_set1.sample(n=3,replace=False)
for i in train_set10.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set11 = train_set1.sample(n=3,replace=False)
for i in train_set11.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set12 = train_set1.sample(n=3,replace=False)
for i in train_set12.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set13 = train_set1.sample(n=3,replace=False)
for i in train_set13.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set14 = train_set1.sample(n=3,replace=False)
for i in train_set14.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set15 = train_set1.sample(n=3,replace=False)
for i in train_set15.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set16 = train_set1.sample(n=3,replace=False)
for i in train_set16.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set17 = train_set1.sample(n=3,replace=False)
for i in train_set17.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set18 = train_set1.sample(n=4,replace=False)
for i in train_set18.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
    
train_set19 = train_set1.sample(n=4,replace=False)
for i in train_set19.index:
    print(i)
    train_set1 = train_set1.drop(i,axis=0)
#2
train_set20 = train_set2.sample(n=6,replace=False)
for i in train_set20.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set21 = train_set2.sample(n=5,replace=False)
for i in train_set21.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set22 = train_set2.sample(n=5,replace=False)
for i in train_set22.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set23 = train_set2.sample(n=5,replace=False)
for i in train_set23.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set24 = train_set2.sample(n=5,replace=False)
for i in train_set24.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set25 = train_set2.sample(n=5,replace=False)
for i in train_set25.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set26 = train_set2.sample(n=5,replace=False)
for i in train_set26.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set27 = train_set2.sample(n=5,replace=False)
for i in train_set27.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set28 = train_set2.sample(n=5,replace=False)
for i in train_set28.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
    
train_set29 = train_set2.sample(n=5,replace=False)
for i in train_set29.index:
    print(i)
    train_set2 = train_set2.drop(i,axis=0)
#3
train_set30 = train_set3.sample(n=7,replace=False)
for i in train_set30.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set31 = train_set3.sample(n=7,replace=False)
for i in train_set31.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set32 = train_set3.sample(n=7,replace=False)
for i in train_set32.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set33 = train_set3.sample(n=7,replace=False)
for i in train_set33.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set34 = train_set3.sample(n=7,replace=False)
for i in train_set34.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set35 = train_set3.sample(n=7,replace=False)
for i in train_set35.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set36 = train_set3.sample(n=7,replace=False)
for i in train_set36.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set37 = train_set3.sample(n=6,replace=False)
for i in train_set37.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set38 = train_set3.sample(n=6,replace=False)
for i in train_set38.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
    
train_set39 = train_set3.sample(n=6,replace=False)
for i in train_set39.index:
    print(i)
    train_set3 = train_set3.drop(i,axis=0)
#4
train_set40 = train_set4.sample(n=4,replace=False)
for i in train_set40.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set41 = train_set4.sample(n=4,replace=False)
for i in train_set41.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set42 = train_set4.sample(n=4,replace=False)
for i in train_set42.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set43 = train_set4.sample(n=4,replace=False)
for i in train_set43.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set44 = train_set4.sample(n=4,replace=False)
for i in train_set44.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set45 = train_set4.sample(n=4,replace=False)
for i in train_set45.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set46 = train_set4.sample(n=4,replace=False)
for i in train_set46.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set47 = train_set4.sample(n=4,replace=False)
for i in train_set47.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set48 = train_set4.sample(n=5,replace=False)
for i in train_set48.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
    
train_set49 = train_set4.sample(n=4,replace=False)
for i in train_set49.index:
    print(i)
    train_set4 = train_set4.drop(i,axis=0)
#5
train_set50 = train_set5.sample(n=2,replace=False)
for i in train_set50.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set51 = train_set5.sample(n=2,replace=False)
for i in train_set51.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set52 = train_set5.sample(n=3,replace=False)
for i in train_set52.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set53 = train_set5.sample(n=3,replace=False)
for i in train_set53.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set54 = train_set5.sample(n=3,replace=False)
for i in train_set54.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set55 = train_set5.sample(n=3,replace=False)
for i in train_set55.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set56 = train_set5.sample(n=3,replace=False)
for i in train_set56.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set57 = train_set5.sample(n=3,replace=False)
for i in train_set57.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set58 = train_set5.sample(n=3,replace=False)
for i in train_set58.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
    
train_set59 = train_set5.sample(n=3,replace=False)
for i in train_set59.index:
    print(i)
    train_set5 = train_set5.drop(i,axis=0)
#6
train_set60 = train_set6.sample(n=3,replace=False)
for i in train_set60.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set61 = train_set6.sample(n=3,replace=False)
for i in train_set61.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set62 = train_set6.sample(n=4,replace=False)
for i in train_set62.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set63 = train_set6.sample(n=4,replace=False)
for i in train_set63.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set64 = train_set6.sample(n=4,replace=False)
for i in train_set64.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set65 = train_set6.sample(n=4,replace=False)
for i in train_set65.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set66 = train_set6.sample(n=4,replace=False)
for i in train_set66.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set67 = train_set6.sample(n=4,replace=False)
for i in train_set67.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set68 = train_set6.sample(n=4,replace=False)
for i in train_set68.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
    
train_set69 = train_set6.sample(n=4,replace=False)
for i in train_set69.index:
    print(i)
    train_set6 = train_set6.drop(i,axis=0)
# 分裂
k=19
train_set_00,train_label_00=np.split(train_set00,indices_or_sections=(k,),axis=1)
train_set_01,train_label_01=np.split(train_set01,indices_or_sections=(k,),axis=1)
train_set_02,train_label_02=np.split(train_set02,indices_or_sections=(k,),axis=1)
train_set_03,train_label_03=np.split(train_set03,indices_or_sections=(k,),axis=1)
train_set_04,train_label_04=np.split(train_set04,indices_or_sections=(k,),axis=1)
train_set_05,train_label_05=np.split(train_set05,indices_or_sections=(k,),axis=1)
train_set_06,train_label_06=np.split(train_set06,indices_or_sections=(k,),axis=1)
train_set_07,train_label_07=np.split(train_set07,indices_or_sections=(k,),axis=1)
train_set_08,train_label_08=np.split(train_set08,indices_or_sections=(k,),axis=1)
train_set_09,train_label_09=np.split(train_set09,indices_or_sections=(k,),axis=1)

train_set_10,train_label_10=np.split(train_set10,indices_or_sections=(k,),axis=1)
train_set_11,train_label_11=np.split(train_set11,indices_or_sections=(k,),axis=1)
train_set_12,train_label_12=np.split(train_set12,indices_or_sections=(k,),axis=1)
train_set_13,train_label_13=np.split(train_set13,indices_or_sections=(k,),axis=1)
train_set_14,train_label_14=np.split(train_set14,indices_or_sections=(k,),axis=1)
train_set_15,train_label_15=np.split(train_set15,indices_or_sections=(k,),axis=1)
train_set_16,train_label_16=np.split(train_set16,indices_or_sections=(k,),axis=1)
train_set_17,train_label_17=np.split(train_set17,indices_or_sections=(k,),axis=1)
train_set_18,train_label_18=np.split(train_set18,indices_or_sections=(k,),axis=1)
train_set_19,train_label_19=np.split(train_set19,indices_or_sections=(k,),axis=1)

train_set_20,train_label_20=np.split(train_set20,indices_or_sections=(k,),axis=1)
train_set_21,train_label_21=np.split(train_set21,indices_or_sections=(k,),axis=1)
train_set_22,train_label_22=np.split(train_set22,indices_or_sections=(k,),axis=1)
train_set_23,train_label_23=np.split(train_set23,indices_or_sections=(k,),axis=1)
train_set_24,train_label_24=np.split(train_set24,indices_or_sections=(k,),axis=1)
train_set_25,train_label_25=np.split(train_set25,indices_or_sections=(k,),axis=1)
train_set_26,train_label_26=np.split(train_set26,indices_or_sections=(k,),axis=1)
train_set_27,train_label_27=np.split(train_set27,indices_or_sections=(k,),axis=1)
train_set_28,train_label_28=np.split(train_set28,indices_or_sections=(k,),axis=1)
train_set_29,train_label_29=np.split(train_set29,indices_or_sections=(k,),axis=1)

train_set_30,train_label_30=np.split(train_set30,indices_or_sections=(k,),axis=1)
train_set_31,train_label_31=np.split(train_set31,indices_or_sections=(k,),axis=1)
train_set_32,train_label_32=np.split(train_set32,indices_or_sections=(k,),axis=1)
train_set_33,train_label_33=np.split(train_set33,indices_or_sections=(k,),axis=1)
train_set_34,train_label_34=np.split(train_set34,indices_or_sections=(k,),axis=1)
train_set_35,train_label_35=np.split(train_set35,indices_or_sections=(k,),axis=1)
train_set_36,train_label_36=np.split(train_set36,indices_or_sections=(k,),axis=1)
train_set_37,train_label_37=np.split(train_set37,indices_or_sections=(k,),axis=1)
train_set_38,train_label_38=np.split(train_set38,indices_or_sections=(k,),axis=1)
train_set_39,train_label_39=np.split(train_set39,indices_or_sections=(k,),axis=1)

train_set_40,train_label_40=np.split(train_set40,indices_or_sections=(k,),axis=1)
train_set_41,train_label_41=np.split(train_set41,indices_or_sections=(k,),axis=1)
train_set_42,train_label_42=np.split(train_set42,indices_or_sections=(k,),axis=1)
train_set_43,train_label_43=np.split(train_set43,indices_or_sections=(k,),axis=1)
train_set_44,train_label_44=np.split(train_set44,indices_or_sections=(k,),axis=1)
train_set_45,train_label_45=np.split(train_set45,indices_or_sections=(k,),axis=1)
train_set_46,train_label_46=np.split(train_set46,indices_or_sections=(k,),axis=1)
train_set_47,train_label_47=np.split(train_set47,indices_or_sections=(k,),axis=1)
train_set_48,train_label_48=np.split(train_set48,indices_or_sections=(k,),axis=1)
train_set_49,train_label_49=np.split(train_set49,indices_or_sections=(k,),axis=1)

train_set_50,train_label_50=np.split(train_set50,indices_or_sections=(k,),axis=1)
train_set_51,train_label_51=np.split(train_set51,indices_or_sections=(k,),axis=1)
train_set_52,train_label_52=np.split(train_set52,indices_or_sections=(k,),axis=1)
train_set_53,train_label_53=np.split(train_set53,indices_or_sections=(k,),axis=1)
train_set_54,train_label_54=np.split(train_set54,indices_or_sections=(k,),axis=1)
train_set_55,train_label_55=np.split(train_set55,indices_or_sections=(k,),axis=1)
train_set_56,train_label_56=np.split(train_set56,indices_or_sections=(k,),axis=1)
train_set_57,train_label_57=np.split(train_set57,indices_or_sections=(k,),axis=1)
train_set_58,train_label_58=np.split(train_set58,indices_or_sections=(k,),axis=1)
train_set_59,train_label_59=np.split(train_set59,indices_or_sections=(k,),axis=1)

train_set_60,train_label_60=np.split(train_set60,indices_or_sections=(k,),axis=1)
train_set_61,train_label_61=np.split(train_set61,indices_or_sections=(k,),axis=1)
train_set_62,train_label_62=np.split(train_set62,indices_or_sections=(k,),axis=1)
train_set_63,train_label_63=np.split(train_set63,indices_or_sections=(k,),axis=1)
train_set_64,train_label_64=np.split(train_set64,indices_or_sections=(k,),axis=1)
train_set_65,train_label_65=np.split(train_set65,indices_or_sections=(k,),axis=1)
train_set_66,train_label_66=np.split(train_set66,indices_or_sections=(k,),axis=1)
train_set_67,train_label_67=np.split(train_set67,indices_or_sections=(k,),axis=1)
train_set_68,train_label_68=np.split(train_set68,indices_or_sections=(k,),axis=1)
train_set_69,train_label_69=np.split(train_set69,indices_or_sections=(k,),axis=1)

# 结合 0-6-8-1-3-2-9
# 结合 6-8-1-3-2-9-5
# 结合 8-1-3-2-9-5-7
# 结合 1-3-2-9-5-7-4
# 结合 3-2-9-5-7-4-0
# 结合 2-9-5-7-4-0-6
# 结合 9-5-7-4-0-6-8
# 结合 5-7-4-0-6-8-1
# 结合 7-4-0-6-8-1-3
# 结合 4-0-6-8-1-3-2

cross_set_0 = pd.concat([train_set_00,train_set_16,train_set_28,train_set_31,
                         train_set_43,train_set_52,train_set_69])
cross_label_0 = pd.concat([train_label_00,train_label_16,train_label_28,train_label_31,
                         train_label_43,train_label_52,train_label_69])
    
cross_set_1 = pd.concat([train_set_06,train_set_18,train_set_21,train_set_33,
                         train_set_42,train_set_59,train_set_65])
cross_label_1 = pd.concat([train_label_06,train_label_18,train_label_21,train_label_33,
                         train_label_42,train_label_59,train_label_65])
    
cross_set_2 = pd.concat([train_set_08,train_set_11,train_set_23,train_set_32,
                         train_set_49,train_set_55,train_set_67])
cross_label_2 = pd.concat([train_label_08,train_label_11,train_label_23,train_label_32,
                         train_label_49,train_label_55,train_label_67])
    
cross_set_3 = pd.concat([train_set_01,train_set_13,train_set_22,train_set_39,
                         train_set_45,train_set_57,train_set_64])
cross_label_3 = pd.concat([train_label_01,train_label_13,train_label_22,train_label_39,
                         train_label_45,train_label_57,train_label_64])
    
cross_set_4 = pd.concat([train_set_03,train_set_12,train_set_29,train_set_35,
                         train_set_47,train_set_54,train_set_60])
cross_label_4 = pd.concat([train_label_03,train_label_12,train_label_29,train_label_35,
                         train_label_47,train_label_54,train_label_60])
    
cross_set_5 = pd.concat([train_set_02,train_set_19,train_set_25,train_set_37,
                         train_set_44,train_set_50,train_set_66])
cross_label_5 = pd.concat([train_label_02,train_label_19,train_label_25,train_label_37,
                         train_label_44,train_label_50,train_label_66])
    
cross_set_6 = pd.concat([train_set_09,train_set_15,train_set_27,train_set_34,
                         train_set_40,train_set_56,train_set_68])
cross_label_6 = pd.concat([train_label_09,train_label_15,train_label_27,train_label_34,
                         train_label_40,train_label_56,train_label_68])
    
cross_set_7 = pd.concat([train_set_05,train_set_17,train_set_24,train_set_30,
                         train_set_46,train_set_58,train_set_61])
cross_label_7 = pd.concat([train_label_05,train_label_17,train_label_24,train_label_30,
                         train_label_46,train_label_58,train_label_61])
    
cross_set_8 = pd.concat([train_set_07,train_set_14,train_set_20,train_set_36,
                         train_set_48,train_set_51,train_set_63])
cross_label_8 = pd.concat([train_label_07,train_label_14,train_label_20,train_label_36,
                         train_label_48,train_label_51,train_label_63])
    
cross_set_9 = pd.concat([train_set_04,train_set_10,train_set_26,train_set_38,
                         train_set_41,train_set_53,train_set_62])
cross_label_9 = pd.concat([train_label_04,train_label_10,train_label_26,train_label_38,
                         train_label_41,train_label_53,train_label_62])
#交叉选取训练集和验证集
xl_1 = pd.concat([cross_set_0,cross_set_1,cross_set_2,
                  cross_set_3,cross_set_4,cross_set_5,
                  cross_set_6,cross_set_7,cross_set_8])
xll_1 = pd.concat([cross_label_0,cross_label_1,cross_label_2,
                  cross_label_3,cross_label_4,cross_label_5,
                  cross_label_6,cross_label_7,cross_label_8]) # 训练label
    
xl_2 = pd.concat([cross_set_0,cross_set_1,cross_set_2,
                  cross_set_3,cross_set_4,cross_set_5,
                  cross_set_6,cross_set_7,cross_set_9])
xll_2 = pd.concat([cross_label_0,cross_label_1,cross_label_2,
                  cross_label_3,cross_label_4,cross_label_5,
                  cross_label_6,cross_label_7,cross_label_9])
    
xl_3 = pd.concat([cross_set_0,cross_set_1,cross_set_2,
                  cross_set_3,cross_set_4,cross_set_5,
                  cross_set_6,cross_set_8,cross_set_9])
xll_3 = pd.concat([cross_label_0,cross_label_1,cross_label_2,
                  cross_label_3,cross_label_4,cross_label_5,
                  cross_label_6,cross_label_8,cross_label_9])
    
xl_4 = pd.concat([cross_set_0,cross_set_1,cross_set_2,
                  cross_set_3,cross_set_4,cross_set_5,
                  cross_set_7,cross_set_8,cross_set_9])
xll_4 = pd.concat([cross_label_0,cross_label_1,cross_label_2,
                  cross_label_3,cross_label_4,cross_label_5,
                  cross_label_7,cross_label_8,cross_label_9])
    
xl_5 = pd.concat([cross_set_0,cross_set_1,cross_set_2,
                  cross_set_3,cross_set_4,cross_set_6,
                  cross_set_7,cross_set_8,cross_set_9])
xll_5 = pd.concat([cross_label_0,cross_label_1,cross_label_2,
                  cross_label_3,cross_label_4,cross_label_6,
                  cross_label_7,cross_label_8,cross_label_9])
    
xl_6 = pd.concat([cross_set_0,cross_set_1,cross_set_2,
                  cross_set_3,cross_set_5,cross_set_6,
                  cross_set_7,cross_set_8,cross_set_9])
xll_6 = pd.concat([cross_label_0,cross_label_1,cross_label_2,
                  cross_label_3,cross_label_5,cross_label_6,
                  cross_label_7,cross_label_8,cross_label_9])
    
xl_7 = pd.concat([cross_set_0,cross_set_1,cross_set_2,
                  cross_set_4,cross_set_5,cross_set_6,
                  cross_set_7,cross_set_8,cross_set_9])
xll_7 = pd.concat([cross_label_0,cross_label_1,cross_label_2,
                  cross_label_4,cross_label_5,cross_label_6,
                  cross_label_7,cross_label_8,cross_label_9])
    
xl_8 = pd.concat([cross_set_0,cross_set_1,cross_set_3,
                  cross_set_4,cross_set_5,cross_set_6,
                  cross_set_7,cross_set_8,cross_set_9])
xll_8 = pd.concat([cross_label_0,cross_label_1,cross_label_3,
                  cross_label_4,cross_label_5,cross_label_6,
                  cross_label_7,cross_label_8,cross_label_9])
    
xl_9 = pd.concat([cross_set_0,cross_set_2,cross_set_3,
                  cross_set_4,cross_set_5,cross_set_6,
                  cross_set_7,cross_set_8,cross_set_9])
xll_9 = pd.concat([cross_label_0,cross_label_2,cross_label_3,
                  cross_label_4,cross_label_5,cross_label_6,
                  cross_label_7,cross_label_8,cross_label_9])
    
xl_10 = pd.concat([cross_set_9,cross_set_1,cross_set_2,
                  cross_set_3,cross_set_4,cross_set_5,
                  cross_set_6,cross_set_7,cross_set_8])
xll_10 = pd.concat([cross_label_9,cross_label_1,cross_label_2,
                  cross_label_3,cross_label_4,cross_label_5,
                  cross_label_6,cross_label_7,cross_label_8])
    
classifier = svm.SVC(C=2, kernel='rbf', gamma=0.05,decision_function_shape='ovr', probability=True)  # ovr:一对多策略
classifier.fit(xl_1, xll_1.values.ravel()) 
classifier.fit(xl_2, xll_2.values.ravel())  
classifier.fit(xl_3, xll_3.values.ravel()) 
classifier.fit(xl_4, xll_4.values.ravel()) 
classifier.fit(xl_5, xll_5.values.ravel()) 
classifier.fit(xl_6, xll_6.values.ravel()) 
classifier.fit(xl_7, xll_7.values.ravel()) 
classifier.fit(xl_8, xll_8.values.ravel()) 
classifier.fit(xl_9, xll_9.values.ravel()) 
classifier.fit(xl_10, xll_10.values.ravel()) 
#交叉验证
scores = cross_val_score(classifier, x, y, cv=10)
scores.mean() 
print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))
xlj = []
csj = []
xlj.append(classifier.score(xl_1,xll_1))
###################################################################### 神经网络
# 数据准备
data = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\3\\17+18+21.csv', engine='python',encoding='UTF-8-sig')
data = data.set_index('Sample')
data = data.iloc[:,:20]
# data = data.drop('Sample',axis=1)
# 确保在每个group中分成0.7 0.3
g0 = data[data['kgroups'] == 0]
g1 = data[data['kgroups'] == 1]
g2 = data[data['kgroups'] == 2]
g3 = data[data['kgroups'] == 3]
g4 = data[data['kgroups'] == 4]
g5 = data[data['kgroups'] == 5]
g6 = data[data['kgroups'] == 6]

split = 19
x0,y0=np.split(g0,indices_or_sections=(split,),axis=1)
train_data0,test_data0,train_label0,test_label0 =train_test_split(x0,y0, random_state=1, train_size=0.7,test_size=0.3)
x1,y1=np.split(g1,indices_or_sections=(split,),axis=1) 
train_data1,test_data1,train_label1,test_label1 =train_test_split(x1,y1, random_state=1, train_size=0.7,test_size=0.3) 
x2,y2=np.split(g2,indices_or_sections=(split,),axis=1) 
train_data2,test_data2,train_label2,test_label2 =train_test_split(x2,y2, random_state=1, train_size=0.7,test_size=0.3) 
x3,y3=np.split(g3,indices_or_sections=(split,),axis=1)
train_data3,test_data3,train_label3,test_label3 =train_test_split(x3,y3, random_state=1, train_size=0.7,test_size=0.3)
x4,y4=np.split(g4,indices_or_sections=(split,),axis=1) 
train_data4,test_data4,train_label4,test_label4 =train_test_split(x4,y4, random_state=1, train_size=0.7,test_size=0.3) 
x5,y5=np.split(g5,indices_or_sections=(split,),axis=1) 
train_data5,test_data5,train_label5,test_label5 =train_test_split(x5,y5, random_state=1, train_size=0.7,test_size=0.3) 
x6,y6=np.split(g6,indices_or_sections=(split,),axis=1) 
train_data6,test_data6,train_label6,test_label6 =train_test_split(x6,y6, random_state=1, train_size=0.7,test_size=0.3) 

frames = [train_data0,train_data1,train_data2,train_data3,train_data4,train_data5,train_data6]
train_data = pd.concat(frames)

frames = [test_data0,test_data1,test_data2,test_data3,test_data4,test_data5,test_data6]
test_data = pd.concat(frames)

frames = [train_label0,train_label1,train_label2,train_label3,train_label4,train_label5,train_label6]
train_label = pd.concat(frames)

frames = [test_label0,test_label1,test_label2,test_label3,test_label4,test_label5,test_label6]
test_label = pd.concat(frames)
frames = [x0,x1,x2,x3,x4,x5,x6]
x = pd.concat(frames)
frames = [y0,y1,y2,y3,y4,y5,y6]
y = pd.concat(frames)

import sklearn.neural_network as snn
s = {}
## 最优参数
for i in range(0,30):
    classifier = snn.MLPClassifier(hidden_layer_sizes=(500,400),activation='relu',solver='sgd',alpha=0.0001,
                                     learning_rate='adaptive',learning_rate_init=0.001,
                                     max_iter=500,random_state=24,shuffle=True)
    classifier.fit(train_data, train_label.values.ravel())
    acc=classifier.score(test_data,test_label) #根据给定数据与标签返回正确率的均值
    s[i] = acc

print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))
classifier.n_features_in_

# 3--hidden_layer_sizes=(500,400) i=24 alpha=0.0001 0.94964
# 
# 

#结果写出
print('train_decision_function:\n',classifier.decision_function(train_data)) 
print('predict_result:\n',classifier.predict(train_data))
len(classifier.predict(train_data))

train_data.info()
train_data.iloc[0:2,18:20]
train_data.insert(22,'Group',classifier.predict(train_data))
train_data.insert(0,'Sample',train_data.index)
train_data.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22_train.csv',index=0)

test_data.info()
test_data.insert(22,'Group',classifier.predict(test_data))
test_data.insert(0,'Sample',test_data.index)
test_data.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22_test.csv',index=0)

## 与kmeans比较正确率
k = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22k=7.csv', engine='python')
m = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22_result.csv', engine='python',encoding='UTF-8-sig')
k = k.set_index('Sample')
m = m.set_index('Sample')
k = k.iloc[:,22]
m = m.iloc[:,22]
r = pd.concat([k,m],axis=1,sort=True)
r.insert(0,'Sample',r.index)
r = r.drop('TCGA-13-1497',axis=0)

m = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22_result.csv', engine='python',encoding='UTF-8-sig')
list_custom = m['Sample'].values
r['Sample'] = r['Sample'].astype('category')
r['Sample'].cat.reorder_categories(list_custom, inplace=True)
r.sort_values('Sample', inplace=True)
r = r.set_index('Sample',drop=False)
r.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22_compare.csv',index=0)

## km
clin = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452clin_info.csv',engine='python')
c = clin.iloc[:,[0,1,5]]
c = c.set_index('sample',drop=True)

b = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\17+18+21_result.csv',engine='python',encoding='UTF-8-sig')
b = b.set_index('Sample',drop=False)
d = b.iloc[:,:21] 
d = d.set_index('Sample',drop=True)

f = pd.concat([d,c],axis=1)
f.info()
fontt={'color': 'k',
      'size': 25,
      'family': 'Arial'}
fonty={'color': 'k',
      'size': 20,
      'family': 'Arial'}
font={'color': 'k',
      'size': 10,
      'family': 'Arial'}

g = f.iloc[:,[20,21,19]]
kmf = KaplanMeierFitter()
groups = g['Group']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)
ix4 = (groups == 3)
ix5 = (groups == 4)
ix6 = (groups == 5)
ix7 = (groups == 6)

T = g['time']
E = g['event']
dem1 = (g['Group'] == 0)
dem2 = (g['Group'] == 1)
dem3 = (g['Group'] == 2)
dem4 = (g['Group'] == 3)
dem5 = (g['Group'] == 4)
dem6 = (g['Group'] == 5)
dem7 = (g['Group'] == 6)

results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem1],T[dem4],E[dem1],E[dem4],alpha=.99)
results_3 = logrank_test(T[dem1],T[dem5],E[dem1],E[dem5],alpha=.99)
results_4 = logrank_test(T[dem1],T[dem6],E[dem1],E[dem6],alpha=.99)
results_5 = logrank_test(T[dem1],T[dem7],E[dem1],E[dem7],alpha=.99)

results_6 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)
results_7 = logrank_test(T[dem2],T[dem4],E[dem2],E[dem4],alpha=.99)
results_8 = logrank_test(T[dem2],T[dem5],E[dem2],E[dem5],alpha=.99)
results_9 = logrank_test(T[dem2],T[dem6],E[dem2],E[dem6],alpha=.99)
results_10 = logrank_test(T[dem2],T[dem7],E[dem2],E[dem7],alpha=.99)

results_11 = logrank_test(T[dem3],T[dem4],E[dem3],E[dem4],alpha=.99)
results_12 = logrank_test(T[dem3],T[dem5],E[dem3],E[dem5],alpha=.99)
results_13 = logrank_test(T[dem3],T[dem6],E[dem3],E[dem6],alpha=.99)
results_14 = logrank_test(T[dem3],T[dem7],E[dem3],E[dem7],alpha=.99)

results_15 = logrank_test(T[dem4],T[dem5],E[dem4],E[dem5],alpha=.99)
results_16 = logrank_test(T[dem4],T[dem6],E[dem4],E[dem6],alpha=.99)
results_17 = logrank_test(T[dem4],T[dem7],E[dem4],E[dem7],alpha=.99)

results_18 = logrank_test(T[dem5],T[dem6],E[dem5],E[dem6],alpha=.99)
results_19 = logrank_test(T[dem5],T[dem7],E[dem5],E[dem7],alpha=.99)
results_20 = logrank_test(T[dem6],T[dem7],E[dem6],E[dem7],alpha=.99)

kmf.fit(g['time'][ix1], g['event'][ix1], label='Group 0')
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=3)
kmf.fit(g['time'][ix2], g['event'][ix2], label='Group 1')
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#BB00217F',linewidth=3)
kmf.fit(g['time'][ix3], g['event'][ix3], label='Group 2') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#0082807F',linewidth=3)
kmf.fit(g['time'][ix4], g['event'][ix4], label='Group 3') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#E18727FF',linewidth=3)
kmf.fit(g['time'][ix5], g['event'][ix5], label='Group 4') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#EE0000FF',linewidth=3)
kmf.fit(g['time'][ix6], g['event'][ix6], label='Group 5') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#808180FF',linewidth=3)
kmf.fit(g['time'][ix7], g['event'][ix7], label='Group 6') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#631879FF',linewidth=3)              
              
plt.legend(loc=(0.53,0.22),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,7300,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
#plt.title('k=7', fontdict=fontt)
#plt.text(0, 0.14, 'Group 0     vs     Group 1'+"      P_value=%.6f"%results.p_value, fontdict=font)
#plt.text(0, 0.08, 'Group 0     vs     Group 2'+"      P_value=%.6f"%results_1.p_value, fontdict=font)
#plt.text(0, 0, 'Group 0     vs     Group 3'+"      P_value=%.6f"%results_2.p_value, fontdict=font)
#plt.text(0, -0.14, 'Group 0     vs     Group 4'+"      P_value=%.6f"%results_3.p_value, fontdict=font)
#plt.text(0, -0.2, 'Group 0     vs     Group 5'+"      P_value=%.6f"%results_4.p_value, fontdict=font)
#plt.text(0, -0.26, 'Group 0     vs     Group 6'+"      P_value=%.6f"%results_5.p_value, fontdict=font)
#plt.text(0, -0.32, 'Group 1     vs     Group 2'+"      P_value=%.6f"%results_6.p_value, fontdict=font)
#plt.text(0, -0.38, 'Group 1     vs     Group 3'+"      P_value=%.6f"%results_7.p_value, fontdict=font)
#plt.text(0, -0.44, 'Group 1     vs     Group 4'+"      P_value=%.6f"%results_8.p_value, fontdict=font)
#plt.text(0, -0.5, 'Group 1     vs     Group 5'+"      P_value=%.6f"%results_9.p_value, fontdict=font)
#plt.text(0, -0.56, 'Group 1     vs     Group 6'+"      P_value=%.6f"%results_10.p_value, fontdict=font)
#plt.text(0, -0.64, 'Group 2     vs     Group 3'+"      P_value=%.6f"%results_11.p_value, fontdict=font)
#plt.text(0, -0.7, 'Group 2     vs     Group 4'+"      P_value=%.6f"%results_12.p_value, fontdict=font)
#plt.text(0, -0.76, 'Group 2     vs     Group 5'+"      P_value=%.6f"%results_13.p_value, fontdict=font)
#plt.text(0, -0.82, 'Group 2     vs     Group 6'+"      P_value=%.6f"%results_14.p_value, fontdict=font)
#plt.text(0, -0.88, 'Group 3     vs     Group 4'+"      P_value=%.6f"%results_15.p_value, fontdict=font)
#plt.text(0, -0.94, 'Group 3     vs     Group 5'+"      P_value=%.6f"%results_16.p_value, fontdict=font)
#plt.text(0, -1.00, 'Group 3     vs     Group 6'+"      P_value=%.6f"%results_17.p_value, fontdict=font)
#plt.text(0, -1.06, 'Group 4     vs     Group 5'+"      P_value=%.6f"%results_18.p_value, fontdict=font)
#plt.text(0, -1.12, 'Group 4     vs     Group 6'+"      P_value=%.6f"%results_19.p_value, fontdict=font)
#plt.text(0, -1.18, 'Group 5     vs     Group 6'+"      P_value=%.6f"%results_20.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 24)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\17+18+21_7.png',figsize=[10,8],dpi=1000, bbox_inches='tight')
plt.close('all')

###### roc画图
x_nd = x.values
y_nd = y.values
y = label_binarize(y_nd, classes=[0,1,2,3,4,5,6])
n_classes = y.shape[1]
n_samples, n_features = x_nd.shape
train_label_1 = label_binarize(train_label, classes=[0,1,2,3,4,5,6])
test_label_1 = label_binarize(test_label, classes=[0,1,2,3,4,5,6])

###########test
train_data = train_data.drop('Sample',axis=1)
test_data = test_data.drop('Sample',axis=1)

y_score = classifier.fit(train_data,train_label.values.ravel()).predict_proba(test_data)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_label_1[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_label_1.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

###########train
y_score_train = classifier.fit(train_data,train_label.values.ravel()).predict_proba(train_data)

fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()
for i in range(n_classes):
    fpr_train[i], tpr_train[i], _ = roc_curve(train_label_1[:, i], y_score_train[:, i])
    roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])
    
# Compute micro-average ROC curve and ROC area
fpr_train["micro"], tpr_train["micro"], _ = roc_curve(train_label_1.ravel(), y_score_train.ravel())
roc_auc_train["micro"] = auc(fpr_train["micro"], tpr_train["micro"])

all_fpr_train = np.unique(np.concatenate([fpr_train[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr_train = np.zeros_like(all_fpr_train)
for i in range(n_classes):
    mean_tpr_train += interp(all_fpr_train, fpr_train[i], tpr_train[i])
# Finally average it and compute AUC
mean_tpr_train /= n_classes
fpr_train["macro"] = all_fpr_train
tpr_train["macro"] = mean_tpr_train
roc_auc_train["macro"] = auc(fpr_train["macro"], tpr_train["macro"])

###########combine
data = pd.concat([train_data,test_data],axis=0)
data.info()
label = pd.concat([train_label,test_label],axis=0)
label.info()
label = label_binarize(label, classes=[0,1,2,3,4,5,6])

y_score_c = classifier.fit(train_data,train_label.values.ravel()).predict_proba(data)

fpr_c = dict()
tpr_c = dict()
roc_auc_c = dict()
for i in range(n_classes):
    fpr_c[i], tpr_c[i], _ = roc_curve(label[:, i], y_score_c[:, i])
    roc_auc_c[i] = auc(fpr_c[i], tpr_c[i])
    
fpr_c["micro"], tpr_c["micro"], _ = roc_curve(label.ravel(), y_score_c.ravel())
roc_auc_c["micro"] = auc(fpr_c["micro"], tpr_c["micro"])

all_fpr_c = np.unique(np.concatenate([fpr_c[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr_c = np.zeros_like(all_fpr_c)
for i in range(n_classes):
    mean_tpr_c += interp(all_fpr_c, fpr_c[i], tpr_c[i])
# Finally average it and compute AUC
mean_tpr_c /= n_classes
fpr_c["macro"] = all_fpr_c
tpr_c["macro"] = mean_tpr_c
roc_auc_c["macro"] = auc(fpr_c["macro"], tpr_c["macro"])

#micro
lw = 2
plt.figure(figsize=(8,8))
plt.plot(fpr["micro"], tpr["micro"],
         label='               {0:0.4f}'
               ''.format(roc_auc["micro"]),
         color='navy', linestyle='-', linewidth=8)

plt.plot(fpr_train["micro"], tpr_train["micro"],
         label='               {0:0.4f}'
               ''.format(roc_auc_train["micro"]),
         color='#7E6148FF', linestyle='-', linewidth=8)

plt.plot(fpr_c["micro"], tpr_c["micro"],
         label='               {0:0.4f}'
               ''.format(roc_auc_c["micro"]),
         color='#F64B35FF', linestyle='-', linewidth=8)
 
plt.plot([0, 1], [0, 1], 'k--', lw=5)
plt.xlim([-0.02, 1.5])
plt.ylim([-0.02, 1.05])
plt.legend(prop={'family' : 'Arial', 'size'   : 38},handletextpad=0.5,frameon=False,labelspacing=0.1,loc=(0.15,-0.03))
plt.tick_params(width=6)
plt.tick_params(length=6)

ax=plt.gca()
ax.spines['bottom'].set_linewidth('4')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('4')
ax.spines['right'].set_linewidth('0')
# plt.title('k=3 top15',fontdict=fontt)
# plt.xlabel('False Positive Rate',fontdict=fonty)
# plt.ylabel('True Positive Rate',fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 0)
plt.xticks(fontproperties = 'Arial', size = 40,rotation=45)
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22iroc.png',dpi=1000, bbox_inches = 'tight')
plt.close('all')

#macro
lw = 2
plt.figure(figsize=(8,8))
plt.plot(fpr["macro"], tpr["macro"],
         label='               {0:0.4f}'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle='-', linewidth=8)

plt.plot(fpr_train["macro"], tpr_train["macro"],
         label='               {0:0.4f}'
               ''.format(roc_auc_train["macro"]),
         color='#7E6148FF', linestyle='-', linewidth=8)

plt.plot(fpr_c["macro"], tpr_c["macro"],
         label='               {0:0.4f}'
               ''.format(roc_auc_c["macro"]),
         color='#F64B35FF', linestyle='-', linewidth=8)
 
plt.plot([0, 1], [0, 1], 'k--', lw=5)
plt.xlim([-0.02, 1.5])
plt.ylim([-0.02, 1.05])
plt.legend(prop={'family' : 'Arial', 'size'   : 38},handletextpad=0.5,frameon=False,labelspacing=0.1,loc=(0.15,-0.03))
plt.tick_params(width=6)
plt.tick_params(length=6)

ax=plt.gca()
ax.spines['bottom'].set_linewidth('4')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('4')
ax.spines['right'].set_linewidth('0')
# plt.title('k=3 top15',fontdict=fontt)
# plt.xlabel('False Positive Rate',fontdict=fonty)
# plt.ylabel('True Positive Rate',fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 0)
plt.xticks(fontproperties = 'Arial', size = 40,rotation=45)
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22aroc.png',dpi=1000, bbox_inches = 'tight')
plt.close('all')

###################################### k=4
clin = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452clin_info.csv',engine='python')
c = clin.iloc[:,[0,1,5]]
c = c.set_index('sample',drop=True)

b = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\17+18+21_result.csv',engine='python',encoding='UTF-8-sig')
b = b.set_index('Sample',drop=False)
d = b.iloc[:,:22] 
d = d.set_index('Sample',drop=True)

f = pd.concat([d,c],axis=1)
f.info()
fontt={'color': 'k',
      'size': 25,
      'family': 'Arial'}
fonty={'color': 'k',
      'size': 20,
      'family': 'Arial'}
font={'color': 'k',
      'size': 10,
      'family': 'Arial'}

g = f.iloc[:,[21,22,20]]
kmf = KaplanMeierFitter()
groups = g['cgroup']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)
ix4 = (groups == 3)

T = g['time']
E = g['event']
dem1 = (g['cgroup'] == 0)
dem2 = (g['cgroup'] == 1)
dem3 = (g['cgroup'] == 2)
dem4 = (g['cgroup'] == 3)
results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem1],T[dem4],E[dem1],E[dem4],alpha=.99)
results_3 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)
results_4 = logrank_test(T[dem2],T[dem4],E[dem2],E[dem4],alpha=.99)
results_5 = logrank_test(T[dem3],T[dem4],E[dem3],E[dem4],alpha=.99)

kmf.fit(g['time'][ix1], g['event'][ix1], label='Group 0')
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=3)
kmf.fit(g['time'][ix2], g['event'][ix2], label='Group 1')
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#BB00217F',linewidth=3)
kmf.fit(g['time'][ix3], g['event'][ix3], label='Group 2+4+5+6') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#0082807F',linewidth=3)
kmf.fit(g['time'][ix4], g['event'][ix4], label='Group 3') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#E18727FF',linewidth=3)

plt.legend(loc=(0.48,0.5),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,3900,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
#plt.title('k=4', fontdict=fontt)
#plt.text(0, 0.14, 'Group 0     vs     Group 1'+"      P_value=%.6f"%results.p_value, fontdict=font)
#plt.text(0, 0.08, 'Group 0     vs     Group 2'+"      P_value=%.6f"%results_1.p_value, fontdict=font)
#plt.text(0, 0, 'Group 0     vs     Group 3'+"      P_value=%.6f"%results_2.p_value, fontdict=font)
#plt.text(0, -0.14, 'Group 1     vs     Group 2'+"      P_value=%.6f"%results_3.p_value, fontdict=font)
#plt.text(0, -0.2, 'Group 1     vs     Group 3'+"      P_value=%.6f"%results_4.p_value, fontdict=font)
#plt.text(0, -0.26, 'Group 2     vs     Group 3'+"      P_value=%.6f"%results_5.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 0)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\17+18+21.png',figsize=[10,8],dpi=1000, bbox_inches='tight')
plt.close('all')




