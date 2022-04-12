import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import orangecontrib.associate.fpgrowth as oaf
import functools
import pytorch3d
import pytorch_geometric
import dgl
import networkx
from scipy import fft
import torch.nn as nn


plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

df = pd.read_csv('Wine Reviews/winemag-data_first150k.csv')
df.head()

df = df.drop(df.columns.values.tolist()[0],axis=1)
df = df.drop(df.columns.values.tolist()[1],axis=1)
df.info()
df.head()


df['price'].hist(bins=20)
plt.show()

def createC1(dataSet):
    C1 = []
    for transaction in np.array(dataSet):
        for item in transaction:
            if [item] not in C1:
                C1.append( [item] )
    C1.sort()
    return list(map( frozenset, C1 ))

def scanD( D, Ck, minSupport ):
    ssCnt = {}
    for tid in D:
        if Ck is not None:
            for can in Ck:
                if can.issubset( tid ):
                    ssCnt[ can ] = ssCnt.get( can, 0) + 1
    numItems = float( len( D ) )
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[ key ] / numItems
        if support >= minSupport:
            retList.insert( 0, key )
            supportData[ key ] = support
    return retList, supportData
import torch.nn as nn

class PairNorm(nn.Module):
    def __init__(self, mode='PN-SCS', scale=1.0):

        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class DynamicPairNorm(nn.Module):

    def __init__(self):
        super(DynamicPairNorm, self).__init__()
    
    def __TransFeauture(self):



def aprioriGen( Lk, k ):
    retList = []
    lenLk = len( Lk )
    for i in range( lenLk ):
        for j in range( i + 1, lenLk ):
            L1 = list( Lk[ i ] )[ : k - 2 ]
            L2 = list( Lk[ j ] )[ : k - 2 ]
            L1.sort();L2.sort()    
            if L1==L2:
                retList.append( Lk[ i ] | Lk[ j ] ) 
    return retList

def apriori( dataSet, minSupport = 0.5 ):
    C1 = createC1( dataSet )
    D =list( map( set, dataSet ))
    L1, suppData = scanD( D, C1, minSupport )
    L = [ L1 ]
    k = 2
    
    while ( len( L[ k - 2 ] ) > 0 ):
        Ck = aprioriGen( L[ k - 2 ], k )
        Lk, supK = scanD( D, Ck, minSupport )
        suppData.update( supK )
        L.append( Lk )
        k += 1
    return L, suppData
myDat = list(map(set,np.array(df)))
L, suppData = apriori(myDat, 0.5)
returnRules = []
for i in df:
    temStr = ''
    for j in i[0]:   
        temStr = temStr+str(L[j])+' & '
    temStr = temStr[:-3]
    temStr = temStr + ' ==> '
    for j in i[1]:
        temStr = temStr+L[j]+' & '
    temStr = temStr[:-3]
    returnRules.append([temStr, round(i[2]/N, 4), round(i[3], 4)])
    temStr = temStr + ';' +'\t'+str(i[2])+ ';' +'\t'+str(i[3])
df = pd.DataFrame(returnRules, columns=['关联规则', '支持度', '置信度']) 
df = df.sort_values('置信度',ascending=False)
df.index = range(len(df))
