#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean

#import seaborn as sns

import itertools
from itertools import product
import timeit
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

## for LaTeX typefont
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

## for another LaTeX typefont
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# rc('text', usetex = True)

print("done")


# In[2]:


data_path = "/alina-data1/Zara/TCell_Receptor/Dataset/"
data_name = "splitted_sequences_2"

seq_data = np.loadtxt(data_path+ data_name + ".npy",dtype=str)


def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers



# # Double Spaced kmers

# In[8]:


gmer_length = 9 # 9
spaced_kmer_length = 4 # 6
unique_seq_kmers_final_list = [''.join(c) for c in product('ABCDEFGHIJKLMNOPQRSTUVWXYZ', repeat=spaced_kmer_length)]  

start = timeit.default_timer()
frequency_vector = []

for seq_ind in range(len(seq_data)):
    #print("Double Spaced kmers => index: ",seq_ind,"/",len(seq_data))
    if seq_ind%1000==0:
        print("index: ",seq_ind,"/",len(seq_data))
    se_temp = seq_data[seq_ind]
    gmers_list = build_kmers(se_temp,gmer_length)


    #extract spaced kmers
    spaced_kmers = []
    for i in range(len(gmers_list)):
        temp_val = gmers_list[i]
        spaced_kmers.append(temp_val[0:spaced_kmer_length])

    #create dictionary
    idx = pd.Index(spaced_kmers) # creates an index which allows counting the entries easily
    # print('Here are all of the viral species in the dataset: \n', len(idx),"entries in total")
    aq = idx.value_counts()
    counter_tmp = aq.values
    gmers_tmp = aq.index
    # counter_tmp,gmers_tmp


    #create frequency vector
    #cnt_check2 = 0
    listofzeros = [0] * len(unique_seq_kmers_final_list)
    for ii in range(len(gmers_tmp)):
        seq_tmp = gmers_tmp[ii]
    #     listofzeros = [0] * len(unique_seq_kmers_final_list)
    #     for j in range(len(seq_tmp)):
        ind_tmp = unique_seq_kmers_final_list.index(seq_tmp)
        listofzeros[ind_tmp] = counter_tmp[ii]
    frequency_vector.append(listofzeros)


stop = timeit.default_timer()
print("Spaced k-mers Embedding Generation Time : ", stop - start) 

np.save(data_path + "Spaced_kmers_g_" + str(gmer_length) + "_k_" + str(spaced_kmer_length) + "_" + data_name + ".npy",frequency_vector)


