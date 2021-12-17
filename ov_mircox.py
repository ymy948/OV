# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:49:28 2021

@author: DELL
"""
import pandas as pd
import numpy as np
targets = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\diff_list.csv',engine='python')
p_value = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\P_value.csv',engine='python')
hr = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\HR.csv',engine='python')
ci = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\CI.csv',engine='python')


## 在p_value中筛选targets
p_value.iloc[0:2,0:2]
p_value.info()
targets.iloc[1,0:2]
targets.info()
len(p_value.columns)
len(targets.columns)
for i in range(0,56):
    if p_value.columns[i] in targets.columns:
        continue
    else:
        p_value = p_value.drop(p_value.columns[i],axis=1)

p_value.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\P_value_1.csv',index=0)

## 在hr中筛选targets
hr.iloc[0:2,0:2]
hr.info()
for i in range(0,56): # 根据报错一直修改range的第二个值
    if hr.columns[i] in targets.columns:
        continue
    else:
        hr = hr.drop(hr.columns[i],axis=1)

hr.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\HR_1.csv',index=0)

## 在ci中筛选targets
for i in range(0,56): # 根据报错一直修改range的第二个值
    if ci.columns[i] in targets.columns:
        continue
    else:
        ci = ci.drop(ci.columns[i],axis=1)

ci.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\CI_1.csv',index=0)

## 生存分析画图
# 数据准备 在z-score中筛选生存相关gene
z_score = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\cox_data.csv',engine='python')
z_score.info()
z_score.iloc[0:2,0:2]
#改为dot
targets = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\diff_list.csv',engine='python')
mrna = list(z_score.columns)
len(mrna)
for i in range(0,391):
    mrna[i] = mrna[i].replace('-','.')
    mrna[i] = mrna[i].replace('|','.')
    mrna[i] = mrna[i].replace('?','.')
for i in range(0,10):
    print(mrna[i])
z_score.columns = mrna

z_score_1 = z_score.iloc[:,:389]
z_score_1.info()
z_score_1.iloc[0:2,0:2]
for i in range(0,56): # 根据报错一直修改range的第二个值388
    if z_score_1.columns[i] in targets.columns:
        continue
    else:
        z_score_1 = z_score_1.drop(z_score_1.columns[i],axis=1)

z_score_1.insert(0,'Gene',z_score['Gene'])         
z_score_1.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\452samples+56miRNAs.csv',index=0)

# supplymentary table
top = 2
k = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\km\\7k=%d.csv'%top,engine='python',encoding='UTF-8-sig')
k = k.iloc[:,]





















