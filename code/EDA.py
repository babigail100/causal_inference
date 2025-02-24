import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
dpq = pd.read_csv(r'.\data\DPQ_L.csv')
paq = pd.read_csv(r'.\data\PAQ_L.csv')
dpq.head()
paq.head()
dpq.shape
paq.shape

comb = pd.merge(dpq, paq, on='SEQN',how='outer')
comb.head()
comb.shape

dpq_merge = dpq.merge(paq, on='SEQN',how='left')
paq_merge = paq.merge(dpq, on='SEQN',how='left')

plt.figure(figsize=(10,6))
sns.heatmap(comb.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(paq_merge.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(dpq_merge.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.show()

sum(dpq['DPQ010'].isnull())
sum(comb['DPQ010'].isnull())

'''
bfs = pd.read_csv(r'.\data\LLCP2023.ASC', delimiter="\t")
bfs.head()

bfs = pd.read_fwf(r'.\data\LLCP2023.ASC')
bfs.head()

msno.matrix(comb)
plt.show()
'''