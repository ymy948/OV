# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:22:29 2021

@author: DELL
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:20:41 2021

@author: DELL
"""
import pandas as pd
import numpy as np
from lifelines.datasets import load_waltons
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
fontt={'color': 'k',
      'size': 25,
      'family': 'Arial'}
fonty={'color': 'k',
      'size': 24,
      'family': 'Arial'}
font={'color': 'k',
      'size': 10,
      'family': 'Arial'}
## 平均值
c = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\452clin_info.csv',engine='python')
c.info()
cc = c.iloc[:,[0,1,5]]
cc.columns = ['index','time','status']
cc = cc.set_index('index')
a = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\452samples+56miRNAs.csv',engine='python')
b = a.set_index('Gene')
b.info()
b = pd.concat([cc,b],axis=1)
i=2
q2 = {}
for i in range(2,58):
    print(b.columns[i])
    c = b.iloc[:,[0,1,i]] #i=2
    c = c.sort_values(by=b.columns[i],ascending=False)
    c_h = c[0:225]
    c_h['group'] = 'High'
    c_l = c[225:452]
    c_l['group'] = 'Low'
    
    d = pd.concat([c_h,c_l],axis=0)
    e = d.iloc[:,[0,1,3]]
    kmf = KaplanMeierFitter()
    groups = e['group']
    ix1 = (groups == 'High')
    ix2 = (groups == 'Low')
    
    T = e['time']
    E = e['status']
    dem1 = (e['group'] == 'High')
    dem2 = (e['group'] == 'Low')
    results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
    
    q2[b.columns[i]] = '%.4f'%results.p_value
    
    kmf.fit(e['time'][ix1], e['status'][ix1], label='High')
    ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F')
    kmf.fit(e['time'][ix2], e['status'][ix2], label='Low')
    ax = kmf.plot(show_censors=True,ax=ax,ci_show=False,color='#BB00217F')
    
    plt.legend(prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
    plt.tick_params(width=2)
    ax.spines['bottom'].set_linewidth('2')
    ax.spines['top'].set_linewidth('0')
    ax.spines['left'].set_linewidth('2')
    ax.spines['right'].set_linewidth('0')
    plt.ylim(-0.08,1.08)
    # plt.axvline(x=1072,c='k',ls='--',lw=0.5)
    plt.title(b.columns[i], fontdict=fontt)
    plt.text(0, 0.05,"P=%.4f"%results.p_value, fontdict=fonty)
    
    plt.xlabel('Timeline(days)', fontdict=fonty)
    plt.ylabel('Cumulative survival (percentage)', fontdict=fonty)
    plt.yticks(fontproperties = 'Arial', size = 24)
    plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
    plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\2 quartile\\%s.png'%b.columns[i],figsize=[10,8],dpi=1000, bbox_inches='tight')
    plt.close('all')

q22 = pd.DataFrame([q2]).T
    
# a = pd.read_csv('E:\\sirebrowser\\STAD\\miR\\分析\\rpm筛选按p排序.csv',engine='python')
# b = a.set_index('Sample')
# b.info()
# fig = plt.figure(figsize=(10,8))
q4 = {}
for i in range(2,58):
    print(b.columns[i])
    c = b.iloc[:,[0,1,i]] #i=2
    c = c.sort_values(by=b.columns[i],ascending=False)
    c_h = c[0:112]
    c_h['group'] = 'High'
    c_l = c[339:452]
    c_l['group'] = 'Low'
    
    d = pd.concat([c_h,c_l],axis=0)
    e = d.iloc[:,[0,1,3]]
    kmf = KaplanMeierFitter()
    groups = e['group']
    ix1 = (groups == 'High')
    ix2 = (groups == 'Low')
    
    T = e['time']
    E = e['status']
    dem1 = (e['group'] == 'High')
    dem2 = (e['group'] == 'Low')
    results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
    
    q4[b.columns[i]] = '%.4f'%results.p_value
    
    kmf.fit(e['time'][ix1], e['status'][ix1], label='High')
    ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F')
    kmf.fit(e['time'][ix2], e['status'][ix2], label='Low')
    ax = kmf.plot(show_censors=True,ax=ax,ci_show=False,color='#BB00217F')
    
    plt.legend(prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
    plt.tick_params(width=2)
    ax.spines['bottom'].set_linewidth('2')
    ax.spines['top'].set_linewidth('0')
    ax.spines['left'].set_linewidth('2')
    ax.spines['right'].set_linewidth('0')
    plt.ylim(-0.08,1.08)
    # plt.axvline(x=1072,c='k',ls='--',lw=0.5)
    plt.title(b.columns[i], fontdict=fontt)
    plt.text(0, 0.05,"P=%.4f"%results.p_value, fontdict=fonty)
    
    plt.xlabel('Timeline(days)', fontdict=fonty)
    plt.ylabel('Cumulative survival (percentage)', fontdict=fonty)
    plt.yticks(fontproperties = 'Arial', size = 24)
    plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
    plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\4 quartile\\%s.png'%b.columns[i],figsize=[10,8],dpi=1000, bbox_inches='tight')
    plt.close('all')
    
q44 = pd.DataFrame([q4]).T
p_value_56 = pd.concat([q22,q44],axis=1)
p_value_56.columns = ['2 quartile','4 quartile']
p_value_56.insert(0,'miRNA',p_value_56.index)
p_value_56.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\p_value_56.csv',index=0)

####################### 在22个re-rank
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

d = pd.read_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22k=7.csv',engine='python')
d = d.set_index('Sample',drop=True)
d = d.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

k = 7
iteration = 500 
model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration,random_state=0) 
model.fit(d) 
e = pd.concat([d, pd.Series(model.labels_, index = d.index)], axis = 1) 
e.columns = list(d.columns) + [u'kgroups']
e.iloc[0:2,0:3]
e.insert(0,'Sample',d.index)
e.to_csv('E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\3\\17+18+21.csv',index=0)


d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\4\\17+18+19+21.csv',engine='python')
d = pd.read_csv(r'E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\22k=7.csv',engine='python')
e = d.set_index('Sample',drop=False)
f = pd.concat([e,c],axis=1)

g = f.iloc[:,[25,26,23]]
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

# #BB00217F深红 #E18727FF黄色 #808180FF深灰色 #631879FF紫色 #EE0000FF橘色 #0082807F绿色 #3B49927F蓝灰色
plt.rcParams['figure.figsize'] = (4,3)
kmf.fit(g['time'][ix1], g['event'][ix1], label='Group 0')
ax = kmf.plot(show_censors=True,ci_show=False,color='#EE0000FF',linewidth=2)
kmf.fit(g['time'][ix2], g['event'][ix2], label='Group 1')
ax = kmf.plot(ax=ax,show_censors=True,ci_show=False,color='#0082807F',linewidth=2)
kmf.fit(g['time'][ix3], g['event'][ix3], label='Group 2') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#BB00217F',linewidth=2)
kmf.fit(g['time'][ix4], g['event'][ix4], label='Group 3') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#E18727FF',linewidth=2)
kmf.fit(g['time'][ix5], g['event'][ix5], label='Group 4') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#631879FF',linewidth=2)
kmf.fit(g['time'][ix6], g['event'][ix6], label='Group 5') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#808180FF',linewidth=2)
kmf.fit(g['time'][ix7], g['event'][ix7], label='Group 6') 
ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F',linewidth=2)  
                      
plt.legend(loc=(0.62,0.3),prop={'family' : 'Arial', 'size'   : 14},handletextpad=0.5,frameon=False,labelspacing=0.1)
ax.get_legend().remove()
plt.tick_params(width=2)
ax.spines['bottom'].set_linewidth('2')
ax.spines['top'].set_linewidth('0')
ax.spines['left'].set_linewidth('2')
ax.spines['right'].set_linewidth('0')
plt.xlim(-230,6300,500)
plt.ylim(-0.08,1.08,0.2)
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
plt.xticks(fontproperties = 'Arial', size = 14,rotation=45)
plt.savefig(fname='E:\\sirebrowser\\OV\\miRNA\\分析\\re-rank by km\\16+6.png',dpi=1200, bbox_inches='tight')
plt.close('all')

