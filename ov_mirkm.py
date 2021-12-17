# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:21:13 2021

@author: DELL
"""
import pandas as pd
import numpy as np
import lifelines
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lifelines.statistics import logrank_test
clin = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452clin_info.csv',engine='python')
c = clin.iloc[:,[0,1,5]]
c = c.set_index('sample',drop=True)
# 将stage作为feature
s = clin.iloc[:,[0,6]] 
s = s.set_index('sample',drop=True)

## p<0.001(2) p<0.002(7) p<0.006(16) p<0.011(22) p<0.021(32) p<0.031(39) p<0.041(49) p<0.051(56)
b = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\rpm按p排序.csv',engine='python')
b = b.set_index('Sample',drop=False)
d = b.iloc[:,:23] 
#d.iloc[0:2,412:415]
d = d.set_index('Sample',drop=True)

#加stage
d = pd.concat([d,s],axis=1)
d = d.replace(np.nan,'null')
d = d[d['stage'] != 'null']
d = d.drop('stage',axis=1)
# 对stage编码
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
#ohe.fit([[2],[3],[4]])
#re = ohe.transform([[2],[3],[4]]).toarray()
#re[1]
## kmeans
k = 7
iteration = 500 
# data = pd.read_csv('E:\\sirebrowser\\STAD\\miR\\分析\\26k=3.csv', engine='python',index_col = 'Sample') 
# data.info()
# data = data.iloc[:,0:25]
model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) 
model.fit(d) 

## KPrototypes
from kmodes.kprototypes import KPrototypes
k = 7
iteration = 500 
model = KPrototypes(n_clusters=k, max_iter=iteration, n_init=10, n_jobs=4,random_state=0)
model.fit(d, categorical=[49]) 

# 输出
e = pd.concat([d, pd.Series(model.labels_, index = d.index)], axis = 1) 
e.columns = list(d.columns) + [u'kgroups']
e.iloc[0:2,0:3]
e.insert(0,'Sample',d.index)
e.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\km\\7k=2.csv',index=0) #保存结果


# 加生存信息
d = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\km\\7k=3.csv',engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

# 画图
fontt={'color': 'k',
      'size': 25,
      'family': 'Arial'}
fonty={'color': 'k',
      'size': 20,
      'family': 'Arial'}
font={'color': 'k',
      'size': 10,
      'family': 'Arial'}

############# k=2
top = 22
gg = 25
tt = 26
ee = 23

d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\km\\%dk=2.csv'%top,engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

g = f.iloc[:,[gg,tt,ee]]
#g = g.drop('TCGA-13-1492',axis=0)

kmf = KaplanMeierFitter()
groups = g['kgroups']
ix1 = (groups == 0)
ix2 = (groups == 1)

T = g['time']
E = g['event']
dem1 = (g['kgroups'] == 0)
dem2 = (g['kgroups'] == 1)
results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)

plt.rcParams['figure.figsize'] = 4,3
kmf.fit(g['time'][ix1], g['event'][ix1], label='Group 0')
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=3)
kmf.fit(g['time'][ix2], g['event'][ix2], label='Group 1')
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#BB00217F',linewidth=3)

plt.legend(loc=(0.48,0.7),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
ax.get_legend().remove()
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,7300,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
#plt.title('k=2', fontdict=fontt)
#plt.text(0, 0.05, 'Group 0     vs     Group 1'+"      P_value=%.8f"%results.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 24)
plt.xticks(fontproperties = 'Arial', size = 0,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname=r'E:\\sirebrowser\\OV\\miRNA\\分析\\4groupkm\\%dk=2.png'%top,figsize=[5,4],dpi=1200, bbox_inches='tight')
plt.close('all')

###################################### k=3
d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\km\\%dk=3.csv'%top,engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

g = f.iloc[:,[gg,tt,ee]]
kmf = KaplanMeierFitter()
groups = g['kgroups']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)

T = g['time']
E = g['event']
dem1 = (g['kgroups'] == 0)
dem2 = (g['kgroups'] == 1)
dem3 = (g['kgroups'] == 2)
results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)

plt.rcParams['figure.figsize'] = 4,3
kmf.fit(g['time'][ix1], g['event'][ix1], label='Group 0')
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=3)
kmf.fit(g['time'][ix2], g['event'][ix2], label='Group 1')
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#BB00217F',linewidth=3)
kmf.fit(g['time'][ix3], g['event'][ix3], label='Group 2') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#0082807F',linewidth=3)

plt.legend(loc=(0.48,0.6),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
ax.get_legend().remove()
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,7300,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
#plt.title('k=3', fontdict=fontt)
#plt.text(0, 0.05, 'Group 0     vs     Group 1'+"      P_value=%.8f"%results.p_value, fontdict=font)
#plt.text(0, 0, 'Group 0     vs     Group 2'+"      P_value=%.8f"%results_1.p_value, fontdict=font)
#plt.text(0, -0.05, 'Group 1     vs     Group 2'+"      P_value=%.8f"%results_2.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 0)
plt.xticks(fontproperties = 'Arial', size = 0,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname=r'E:\\sirebrowser\\OV\\miRNA\\分析\\4groupkm\\%dk=3.png'%top,figsize=[5,4],dpi=1200, bbox_inches='tight')
plt.close('all')

###################################### k=4
d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\km\\%dk=4.csv'%top,engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

g = f.iloc[:,[gg,tt,ee]]
kmf = KaplanMeierFitter()
groups = g['kgroups']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)
ix4 = (groups == 3)

T = g['time']
E = g['event']
dem1 = (g['kgroups'] == 0)
dem2 = (g['kgroups'] == 1)
dem3 = (g['kgroups'] == 2)
dem4 = (g['kgroups'] == 3)
results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem1],T[dem4],E[dem1],E[dem4],alpha=.99)
results_3 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)
results_4 = logrank_test(T[dem2],T[dem4],E[dem2],E[dem4],alpha=.99)
results_5 = logrank_test(T[dem3],T[dem4],E[dem3],E[dem4],alpha=.99)

plt.rcParams['figure.figsize'] = 4,3
kmf.fit(g['time'][ix1], g['event'][ix1], label='Group 0')
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=3)
#kmf.median_survival_time_
kmf.fit(g['time'][ix2], g['event'][ix2], label='Group 1')     
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#BB00217F',linewidth=3)
kmf.fit(g['time'][ix3], g['event'][ix3], label='Group 2') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#0082807F',linewidth=3)
kmf.fit(g['time'][ix4], g['event'][ix4], label='Group 3') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#E18727FF',linewidth=3)

plt.legend(loc=(0.48,0.5),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
ax.get_legend().remove()
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,7300,500)
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
plt.xticks(fontproperties = 'Arial', size = 0,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname=r'E:\\sirebrowser\\OV\\miRNA\\分析\\4groupkm\\%dk=4.png'%top,figsize=[5,4],dpi=1200, bbox_inches='tight')
plt.close('all')

###################################### k=5
d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\km\\%dk=5.csv'%top,engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

g = f.iloc[:,[gg,tt,ee]]
kmf = KaplanMeierFitter()
groups = g['kgroups']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)
ix4 = (groups == 3)
ix5 = (groups == 4)

T = g['time']
E = g['event']
dem1 = (g['kgroups'] == 0)
dem2 = (g['kgroups'] == 1)
dem3 = (g['kgroups'] == 2)
dem4 = (g['kgroups'] == 3)
dem5 = (g['kgroups'] == 4)

results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem1],T[dem4],E[dem1],E[dem4],alpha=.99)
results_3 = logrank_test(T[dem1],T[dem5],E[dem1],E[dem5],alpha=.99)
results_4 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)
results_5 = logrank_test(T[dem2],T[dem4],E[dem2],E[dem4],alpha=.99)
results_6 = logrank_test(T[dem2],T[dem5],E[dem2],E[dem5],alpha=.99)
results_7 = logrank_test(T[dem3],T[dem4],E[dem3],E[dem4],alpha=.99)
results_8 = logrank_test(T[dem3],T[dem5],E[dem3],E[dem5],alpha=.99)
results_9 = logrank_test(T[dem4],T[dem5],E[dem4],E[dem5],alpha=.99)

plt.rcParams['figure.figsize'] = 4,3
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

              
plt.legend(loc=(0.48,0.45),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
ax.get_legend().remove()
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,7300,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
#plt.title('k=5', fontdict=fontt)
#plt.text(0, 0.14, 'Group 0     vs     Group 1'+"      P_value=%.6f"%results.p_value, fontdict=font)
#plt.text(0, 0.08, 'Group 0     vs     Group 2'+"      P_value=%.6f"%results_1.p_value, fontdict=font)
#plt.text(0, 0, 'Group 0     vs     Group 3'+"      P_value=%.6f"%results_2.p_value, fontdict=font)
#plt.text(0, -0.14, 'Group 0     vs     Group 4'+"      P_value=%.6f"%results_3.p_value, fontdict=font)
#plt.text(0, -0.2, 'Group 1     vs     Group 2'+"      P_value=%.6f"%results_4.p_value, fontdict=font)
#plt.text(0, -0.26, 'Group 1     vs     Group 3'+"      P_value=%.6f"%results_5.p_value, fontdict=font)
#plt.text(0, -0.32, 'Group 1     vs     Group 4'+"      P_value=%.6f"%results_6.p_value, fontdict=font)
#plt.text(0, -0.38, 'Group 2     vs     Group 3'+"      P_value=%.6f"%results_7.p_value, fontdict=font)
#plt.text(0, -0.42, 'Group 2     vs     Group 4'+"      P_value=%.6f"%results_8.p_value, fontdict=font)
#plt.text(0, -0.48, 'Group 3     vs     Group 4'+"      P_value=%.6f"%results_9.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 0)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname=r'E:\\sirebrowser\\OV\\miRNA\\分析\\4groupkm\\%dk=5.png'%top,figsize=[5,4],dpi=1200, bbox_inches='tight')
plt.close('all')

###################################### k=6
d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\km\\%dk=6.csv'%top,engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

g = f.iloc[:,[gg,tt,ee]]
kmf = KaplanMeierFitter()
groups = g['kgroups']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)
ix4 = (groups == 3)
ix5 = (groups == 4)
ix6 = (groups == 5)

T = g['time']
E = g['event']
dem1 = (g['kgroups'] == 0)
dem2 = (g['kgroups'] == 1)
dem3 = (g['kgroups'] == 2)
dem4 = (g['kgroups'] == 3)
dem5 = (g['kgroups'] == 4)
dem6 = (g['kgroups'] == 5)

results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem1],T[dem4],E[dem1],E[dem4],alpha=.99)
results_3 = logrank_test(T[dem1],T[dem5],E[dem1],E[dem5],alpha=.99)
results_4 = logrank_test(T[dem1],T[dem6],E[dem1],E[dem6],alpha=.99)

results_5 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)
results_6 = logrank_test(T[dem2],T[dem4],E[dem2],E[dem4],alpha=.99)
results_7 = logrank_test(T[dem2],T[dem5],E[dem2],E[dem5],alpha=.99)
results_8 = logrank_test(T[dem2],T[dem6],E[dem2],E[dem6],alpha=.99)

results_9 = logrank_test(T[dem3],T[dem4],E[dem3],E[dem4],alpha=.99)
results_10 = logrank_test(T[dem3],T[dem5],E[dem3],E[dem5],alpha=.99)
results_11 = logrank_test(T[dem3],T[dem6],E[dem3],E[dem6],alpha=.99)

results_12 = logrank_test(T[dem4],T[dem5],E[dem4],E[dem5],alpha=.99)
results_13 = logrank_test(T[dem4],T[dem6],E[dem4],E[dem6],alpha=.99)
results_14 = logrank_test(T[dem5],T[dem6],E[dem5],E[dem6],alpha=.99)


plt.rcParams['figure.figsize'] = 4,3
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
              
plt.legend(loc=(0.5,0.32),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
ax.get_legend().remove()
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,7300,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
#plt.title('k=6', fontdict=fontt)
#plt.text(0, 0.14, 'Group 0     vs     Group 1'+"      P_value=%.6f"%results.p_value, fontdict=font)
#plt.text(0, 0.08, 'Group 0     vs     Group 2'+"      P_value=%.6f"%results_1.p_value, fontdict=font)
#plt.text(0, 0, 'Group 0     vs     Group 3'+"      P_value=%.6f"%results_2.p_value, fontdict=font)
#plt.text(0, -0.14, 'Group 0     vs     Group 4'+"      P_value=%.6f"%results_3.p_value, fontdict=font)
#plt.text(0, -0.2, 'Group 0     vs     Group 5'+"      P_value=%.6f"%results_4.p_value, fontdict=font)
#plt.text(0, -0.26, 'Group 1     vs     Group 2'+"      P_value=%.6f"%results_5.p_value, fontdict=font)
#plt.text(0, -0.32, 'Group 1     vs     Group 3'+"      P_value=%.6f"%results_6.p_value, fontdict=font)
#plt.text(0, -0.38, 'Group 1     vs     Group 4'+"      P_value=%.6f"%results_7.p_value, fontdict=font)
#plt.text(0, -0.44, 'Group 1     vs     Group 5'+"      P_value=%.6f"%results_8.p_value, fontdict=font)
#plt.text(0, -0.5, 'Group 2     vs     Group 3'+"      P_value=%.6f"%results_9.p_value, fontdict=font)
#plt.text(0, -0.56, 'Group 2     vs     Group 4'+"      P_value=%.6f"%results_10.p_value, fontdict=font)
#plt.text(0, -0.64, 'Group 2     vs     Group 5'+"      P_value=%.6f"%results_11.p_value, fontdict=font)
#plt.text(0, -0.7, 'Group 3     vs     Group 4'+"      P_value=%.6f"%results_12.p_value, fontdict=font)
#plt.text(0, -0.76, 'Group 3     vs     Group 5'+"      P_value=%.6f"%results_13.p_value, fontdict=font)
#plt.text(0, -0.84, 'Group 4     vs     Group 5'+"      P_value=%.6f"%results_14.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 24)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname=r'E:\\sirebrowser\\OV\\miRNA\\分析\\4groupkm\\%dk=6.png'%top,figsize=[5,4],dpi=1200, bbox_inches='tight')
plt.close('all')

###################################### k=7
d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\km\\%dk=7.csv'%top,engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

g = f.iloc[:,[gg,tt,ee]]
kmf = KaplanMeierFitter()
groups = g['kgroups']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)
ix4 = (groups == 3)
ix5 = (groups == 4)
ix6 = (groups == 5)
ix7 = (groups == 6)

T = g['time']
E = g['event']
dem1 = (g['kgroups'] == 0)
dem2 = (g['kgroups'] == 1)
dem3 = (g['kgroups'] == 2)
dem4 = (g['kgroups'] == 3)
dem5 = (g['kgroups'] == 4)
dem6 = (g['kgroups'] == 5)
dem7 = (g['kgroups'] == 6)

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

plt.rcParams['figure.figsize'] = 4,3
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
ax.get_legend().remove()
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
plt.yticks(fontproperties = 'Arial', size = 0)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname=r'E:\\sirebrowser\\OV\\miRNA\\分析\\4groupkm\\%dk=7.png'%top,figsize=[5,4],dpi=1200, bbox_inches='tight')
plt.close('all')

###################################### k=8
d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\km\\%dk=8.csv'%top,engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

g = f.iloc[:,[gg,tt,ee]]
kmf = KaplanMeierFitter()
groups = g['kgroups']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)
ix4 = (groups == 3)
ix5 = (groups == 4)
ix6 = (groups == 5)
ix7 = (groups == 6)
ix8 = (groups == 7)


T = g['time']
E = g['event']
dem1 = (g['kgroups'] == 0)
dem2 = (g['kgroups'] == 1)
dem3 = (g['kgroups'] == 2)
dem4 = (g['kgroups'] == 3)
dem5 = (g['kgroups'] == 4)
dem6 = (g['kgroups'] == 5)
dem7 = (g['kgroups'] == 6)
dem8 = (g['kgroups'] == 7)


results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem1],T[dem4],E[dem1],E[dem4],alpha=.99)
results_3 = logrank_test(T[dem1],T[dem5],E[dem1],E[dem5],alpha=.99)
results_4 = logrank_test(T[dem1],T[dem6],E[dem1],E[dem6],alpha=.99)
results_5 = logrank_test(T[dem1],T[dem7],E[dem1],E[dem7],alpha=.99)
results_6 = logrank_test(T[dem1],T[dem8],E[dem1],E[dem8],alpha=.99)


results_7 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)
results_8 = logrank_test(T[dem2],T[dem4],E[dem2],E[dem4],alpha=.99)
results_9 = logrank_test(T[dem2],T[dem5],E[dem2],E[dem5],alpha=.99)
results_10 = logrank_test(T[dem2],T[dem6],E[dem2],E[dem6],alpha=.99)
results_11 = logrank_test(T[dem2],T[dem7],E[dem2],E[dem7],alpha=.99)
results_12 = logrank_test(T[dem2],T[dem8],E[dem2],E[dem8],alpha=.99)


results_13 = logrank_test(T[dem3],T[dem4],E[dem3],E[dem4],alpha=.99)
results_14 = logrank_test(T[dem3],T[dem5],E[dem3],E[dem5],alpha=.99)
results_15 = logrank_test(T[dem3],T[dem6],E[dem3],E[dem6],alpha=.99)
results_16 = logrank_test(T[dem3],T[dem7],E[dem3],E[dem7],alpha=.99)
results_17 = logrank_test(T[dem3],T[dem8],E[dem3],E[dem8],alpha=.99)


results_18 = logrank_test(T[dem4],T[dem5],E[dem4],E[dem5],alpha=.99)
results_19 = logrank_test(T[dem4],T[dem6],E[dem4],E[dem6],alpha=.99)
results_20 = logrank_test(T[dem4],T[dem7],E[dem4],E[dem7],alpha=.99)
results_21 = logrank_test(T[dem4],T[dem8],E[dem4],E[dem8],alpha=.99)


results_22 = logrank_test(T[dem5],T[dem6],E[dem5],E[dem6],alpha=.99)
results_23 = logrank_test(T[dem5],T[dem7],E[dem5],E[dem7],alpha=.99)
results_24 = logrank_test(T[dem5],T[dem8],E[dem5],E[dem8],alpha=.99)

results_25 = logrank_test(T[dem6],T[dem7],E[dem6],E[dem7],alpha=.99)
results_26 = logrank_test(T[dem6],T[dem8],E[dem6],E[dem8],alpha=.99)
results_27 = logrank_test(T[dem7],T[dem8],E[dem7],E[dem8],alpha=.99)

plt.rcParams['figure.figsize'] = 4,3
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
kmf.fit(g['time'][ix8], g['event'][ix8], label='Group 7') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#008B45FF',linewidth=3)               
              
plt.legend(loc=(0.56,0.08),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
ax.get_legend().remove()
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,7300,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
#plt.title('k=8', fontdict=fontt)
#plt.text(0, 0.14, 'Group 0     vs     Group 1'+"      P_value=%.6f"%results.p_value, fontdict=font)
#plt.text(0, 0.08, 'Group 0     vs     Group 2'+"      P_value=%.6f"%results_1.p_value, fontdict=font)
#plt.text(0, 0, 'Group 0     vs     Group 3'+"      P_value=%.6f"%results_2.p_value, fontdict=font)
#plt.text(0, -0.14, 'Group 0     vs     Group 4'+"      P_value=%.6f"%results_3.p_value, fontdict=font)
#plt.text(0, -0.2, 'Group 0     vs     Group 5'+"      P_value=%.6f"%results_4.p_value, fontdict=font)
#plt.text(0, -0.26, 'Group 0     vs     Group 6'+"      P_value=%.6f"%results_5.p_value, fontdict=font)
#plt.text(0, -0.32, 'Group 0     vs     Group 7'+"      P_value=%.6f"%results_6.p_value, fontdict=font)
#plt.text(0, -0.38, 'Group 1     vs     Group 2'+"      P_value=%.6f"%results_7.p_value, fontdict=font)
#plt.text(0, -0.44, 'Group 1     vs     Group 3'+"      P_value=%.6f"%results_8.p_value, fontdict=font)
#plt.text(0, -0.5, 'Group 1     vs     Group 4'+"      P_value=%.6f"%results_9.p_value, fontdict=font)
#plt.text(0, -0.56, 'Group 1     vs     Group 5'+"      P_value=%.6f"%results_10.p_value, fontdict=font)
#plt.text(0, -0.62, 'Group 1     vs     Group 6'+"      P_value=%.6f"%results_11.p_value, fontdict=font)
#plt.text(0, -0.68, 'Group 1     vs     Group 7'+"      P_value=%.6f"%results_12.p_value, fontdict=font)
#plt.text(0, -0.74, 'Group 2     vs     Group 3'+"      P_value=%.6f"%results_13.p_value, fontdict=font)
#plt.text(0, -0.8, 'Group 2     vs     Group 4'+"      P_value=%.6f"%results_14.p_value, fontdict=font)
#plt.text(0, -0.86, 'Group 2     vs     Group 5'+"      P_value=%.6f"%results_15.p_value, fontdict=font)
#plt.text(0, -0.92, 'Group 2     vs     Group 6'+"      P_value=%.6f"%results_16.p_value, fontdict=font)
#plt.text(0, -0.98, 'Group 2     vs     Group 7'+"      P_value=%.6f"%results_17.p_value, fontdict=font)
#plt.text(0, -1.04, 'Group 3     vs     Group 4'+"      P_value=%.6f"%results_18.p_value, fontdict=font)
#plt.text(0, -1.1, 'Group 3     vs     Group 5'+"      P_value=%.6f"%results_19.p_value, fontdict=font)
#plt.text(0, -1.16, 'Group 3     vs     Group 6'+"      P_value=%.6f"%results_20.p_value, fontdict=font)
#plt.text(0, -1.22, 'Group 3     vs     Group 7'+"      P_value=%.6f"%results_21.p_value, fontdict=font)
#plt.text(0, -1.28, 'Group 4     vs     Group 5'+"      P_value=%.6f"%results_22.p_value, fontdict=font)
#plt.text(0, -1.34, 'Group 4     vs     Group 6'+"      P_value=%.6f"%results_23.p_value, fontdict=font)
#plt.text(0, -1.4, 'Group 4     vs     Group 7'+"      P_value=%.6f"%results_24.p_value, fontdict=font)
#plt.text(0, -1.46, 'Group 5     vs     Group 6'+"      P_value=%.6f"%results_25.p_value, fontdict=font)
#plt.text(0, -1.52, 'Group 5     vs     Group 7'+"      P_value=%.6f"%results_26.p_value, fontdict=font)
#plt.text(0, -1.58, 'Group 6     vs     Group 7'+"      P_value=%.6f"%results_27.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 0)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname=r'E:\\sirebrowser\\OV\\miRNA\\分析\\4groupkm\\%dk=8.png'%top,figsize=[5,4],dpi=1200, bbox_inches='tight')
plt.close('all')
###################################################################################################
###################################################################################################
###################################################################################################
#########################################################3 统计分组信息
## 合并同top不同k
g = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\km\\56k=8.csv',engine='python')
g = g.iloc[:,[0,57]]
g = g.set_index('Sample',drop=True)
g.columns = ['K=8']
#h = g # k=2
h = pd.concat([h,g],axis=1) # k!=2

h.insert(0,'Sample',h.index)
h.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\56k.csv',index=0)

# 统计
h = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\22k.csv',engine='python')
for i in range(0,2):
    print('k=2,group%d'%i,len(h['K=2'][h['K=2'] == i]))
    for j in range(0,3):
        print('k=3,group%d'%j,len(h[(h['K=2'] == i)&(h['K=3'] == j)]))
    print('---')    
print('--------------------') 
for i in range(0,3):
    print('k=3,group%d'%i,len(h['K=3'][h['K=3'] == i]))
    for j in range(0,4):
        print('k=4,group%d'%j,len(h[(h['K=3'] == i)&(h['K=4'] == j)]))
    print('---') 
print('--------------------') 
for i in range(0,4):
    print('k=4,group%d'%i,len(h['K=4'][h['K=4'] == i]))
    for j in range(0,5):
        print('k=5,group%d'%j,len(h[(h['K=4'] == i)&(h['K=5'] == j)]))
    print('---')         
print('--------------------') 
for i in range(0,5):
    print('k=5,group%d'%i,len(h['K=5'][h['K=5'] == i]))
    for j in range(0,6):
        print('k=6,group%d'%j,len(h[(h['K=5'] == i)&(h['K=6'] == j)]))
    print('---') 
print('--------------------')     
for i in range(0,6):
    print('k=6,group%d'%i,len(h['K=6'][h['K=6'] == i]))
    for j in range(0,7):
        print('k=7,group%d'%j,len(h[(h['K=6'] == i)&(h['K=7'] == j)]))
    print('---') 
print('--------------------')     
for i in range(0,7):
    print('k=7,group%d'%i,len(h['K=7'][h['K=7'] == i]))
    for j in range(0,8):
        print('k=8,group%d'%j,len(h[(h['K=7'] == i)&(h['K=8'] == j)]))
    print('---') 
############################################################# 堆积柱状图
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

x=[1,2,3]
classes=['k = 2','k = 3','k = 4']#,'k = 5','k = 6','k = 7','k = 8'
Group7= [0,0,0]
Group6= [0,0,0]
Group5= [0,0,0]
Group4= [0,2,0]
Group3= [0,65,0]
Group2= [0,155,120]
Group1= [230,118,177]
Group0= [222,112,155]

plt.bar(x,Group0,label='Group0',color='#3B49927F')
        
plt.bar(x,Group1,bottom=Group0,color='#BB00217F',label='Group1')
cheng=[Group0[i]+Group1[i] for i in range(len(x))]

plt.bar(x,Group2,bottom=cheng,color='#4320cd80',label='Group2')
cheng1=[Group0[i]+Group1[i]+Group2[i] for i in range(len(x))]

plt.bar(x,Group3,bottom=cheng1,color='#858470FF',label='Group3')
cheng2=[Group0[i]+Group1[i]+Group2[i]+Group3[i] for i in range(len(x))]

plt.bar(x,Group4,bottom=cheng2,color='#F5164dFF',label='Group4')
cheng3=[Group0[i]+Group1[i]+Group2[i]+Group3[i] for i in range(len(x))]

plt.bar(x,Group5,bottom=cheng3,color='#BB8032FF',label='Group5')
cheng4=[Group0[i]+Group1[i]+Group2[i]+Group3[i]+Group4[i] for i in range(len(x))]

plt.bar(x,Group6,bottom=cheng4,color='#BB124dFF',label='Group6')
cheng5=[Group0[i]+Group1[i]+Group2[i]+Group3[i] for i in range(len(x))]

plt.bar(x,Group7,bottom=cheng5,color='#BD75203F',label='Group7')
cheng6=[Group0[i]+Group1[i]+Group2[i]+Group3[i] for i in range(len(x))]


plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('2')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('2')
plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 24)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
plt.ylim([0,500])
plt.xlim([0,20])
plt.title("Top 2", fontdict=fontt) #标题
plt.xticks(x,classes) #xticks() 对应坐标名称
plt.legend(prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)

#plt.grid(axis='y',color='gray',linestyle='--',linewidth=1)
plt.show()


################################################## clinical feature km
clin = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452clin_info.csv',engine='python')
c = clin.iloc[:,[0,1,5,6]]
c = c.set_index('sample',drop=True)
c = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\stage.csv',engine='python')

g = c
kmf = KaplanMeierFitter()
groups = g['stage']
ix1 = (groups == 2)
ix2 = (groups == 3)
ix3 = (groups == 4)

T = g['time']
E = g['event']
dem1 = (g['stage'] == 2)
dem2 = (g['stage'] == 3)
dem3 = (g['stage'] == 4)
results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)

kmf.fit(g['time'][ix1], g['event'][ix1], label='Stage Ⅱ')
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=3)
kmf.fit(g['time'][ix2], g['event'][ix2], label='Stage Ⅲ')
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#BB00217F',linewidth=3)
kmf.fit(g['time'][ix3], g['event'][ix3], label='Stage Ⅳ') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#0082807F',linewidth=3)

plt.legend(loc=(0.48,0.6),prop={'family' : 'SimHei', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,3900,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
plt.title('k=3', fontdict=fontt)
#plt.text(0, 0.05, 'Group 0     vs     Group 1'+"      P_value=%.8f"%results.p_value, fontdict=font)
#plt.text(0, 0, 'Group 0     vs     Group 2'+"      P_value=%.8f"%results_1.p_value, fontdict=font)
#plt.text(0, -0.05, 'Group 1     vs     Group 2'+"      P_value=%.8f"%results_2.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 24)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\km\\2k=3.png',figsize=[10,8],dpi=1000)
plt.close('all')

##################3 moculer features and clinical features combination
c = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\stage.csv',engine='python')
a = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\km\\39k=3.csv',engine='python')
c.columns = ['sample','time','event','stage']
c = c.set_index('sample',drop=True)
b = a.iloc[:,[0,40]]
b.columns = ['sample','kgroups']
b = b.set_index('sample',drop=True)

r = pd.concat([c,b],axis=1)
r.insert(0,'sample',r.index)

r.to_csv('E:\\sirebrowser\\OV\\miRNA\\combine.csv',index=0) 

com = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\combine筛选.csv',engine='python')
com1 = com.iloc[:,[1,2,4]]
g = com1
kmf = KaplanMeierFitter()
groups = g['kgroups']
ix1 = (groups == 0)
ix2 = (groups == 1)
ix3 = (groups == 2)

T = g['time']
E = g['event']
dem1 = (g['kgroups'] == 0)
dem2 = (g['kgroups'] == 1)
dem3 = (g['kgroups'] == 2)
results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
results_1 = logrank_test(T[dem1],T[dem3],E[dem1],E[dem3],alpha=.99)
results_2 = logrank_test(T[dem2],T[dem3],E[dem2],E[dem3],alpha=.99)

kmf.fit(g['time'][ix1], g['event'][ix1], label='Group 0')
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=3)
kmf.fit(g['time'][ix2], g['event'][ix2], label='Group 1')
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#BB00217F',linewidth=3)
kmf.fit(g['time'][ix3], g['event'][ix3], label='Group 2') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#0082807F',linewidth=3)

plt.legend(loc=(0.48,0.6),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,3900,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
plt.title('k=3', fontdict=fontt)
#plt.text(0, 0.05, 'Group 0     vs     Group 1'+"      P_value=%.8f"%results.p_value, fontdict=font)
#plt.text(0, 0, 'Group 0     vs     Group 2'+"      P_value=%.8f"%results_1.p_value, fontdict=font)
#plt.text(0, -0.05, 'Group 1     vs     Group 2'+"      P_value=%.8f"%results_2.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 0)
plt.xticks(fontproperties = 'Arial', size = 0,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\km\\2k=3.png',figsize=[10,8],dpi=1000)
plt.close('all')

##

from kmodes.kprototypes import KPrototypes

import numpy as np
from numpy import random
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


R = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\km\\22k=7heatmap.csv',engine='python')
R = R.set_index('Sample',drop=True)
sns.set()
fig = plt.figure()
sns_plot = sns.heatmap(R)
# fig.savefig("heatmap.pdf", bbox_inches='tight') # 减少边缘空白
plt.show()

####################################结合不同聚类结果
ori = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\km\\22k=7.csv',engine='python')
g4 = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\聚类heatmap444.csv',engine='python')
g7 = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\聚类heatmap777.csv',engine='python')
ori = ori.set_index('Sample',drop=True)
g4 = g4.set_index('Sample',drop=True)
g7 = g7.set_index('Sample',drop=True)
result = pd.concat([ori,g4,g7],axis=1)
result.insert(0,'Sample',result.index)
result.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\different_clustering_result.csv',index=0) #保存结果

###########中位生存天数
clin = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452clin_info.csv',engine='python')
c = clin.iloc[:,[0,1,5]]
c = c.set_index('sample',drop=True)

## 
b = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\rpm筛选排序.csv',engine='python')
b = b.set_index('Sample',drop=False)

f = pd.concat([b,c],axis=1)
f.info()
g = f.iloc[:,[58,59,57]]

kmf = KaplanMeierFitter()
groups = g['group']
ix1 = (groups == 0)
ix2 = (groups == 1)

T = g['time']
E = g['event']
dem1 = (g['group'] == 0)
dem2 = (g['group'] == 1)

kmf.median_survival_time_


results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)

kmf.fit(g['time'][ix1], g['event'][ix1], label='Group 0')
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=3)
kmf.median_survival_time_
kmf.fit(g['time'][ix2], g['event'][ix2], label='Group 1')
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#BB00217F',linewidth=3)
kmf.median_survival_time_

plt.legend(loc=(0.48,0.7),prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
plt.tick_params(width=4)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,3900,500)
plt.ylim(-0.08,1.08)
plt.axvline(x=1825,c='k',ls='--',lw=2)
plt.axvline(x=1095,c='k',ls='--',lw=2)
#plt.title('k=2', fontdict=fontt)
#plt.text(0, 0.05, 'Group 0     vs     Group 1'+"      P_value=%.8f"%results.p_value, fontdict=font)

plt.xlabel(' ')
plt.ylabel(' ', fontdict=fonty)
plt.yticks(fontproperties = 'Arial', size = 24)
plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
# plt.figure(dpi=1000,figsize=(24,20))
# 绘制figure3时改变了组别编号，不影响结果
