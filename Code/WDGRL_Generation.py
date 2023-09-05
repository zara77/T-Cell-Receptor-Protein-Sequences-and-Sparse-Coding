import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
#import RandomBinningFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt 
#%matplotlib inline 
import csv

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

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from numpy import mean
#import seaborn as sns
from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
from adapt.utils import make_classification_da
from adapt.feature_based import WDGRL



print("Packages Loaded!!!")



#Xx = np.load("/alina-data1/sarwan/IEEE_BigData/Dataset/OHE_one_point_9_million_freq_vec.npy",allow_pickle=True)
data_path = "/alina-data1/Zara/TCell_Receptor/Embedding_Generation/"
data_name = "OHE_splitted_sequences_50k_1"
Xx = np.load(data_path + data_name + ".npy")


attribute_data =  np.loadtxt(data_path + "splitted_Labels_50k_1.npy",dtype=str)

attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[","")
    aa_1 = aa.replace("]","")
    aa_2 = aa_1.replace("\'","")
    attr_new.append(aa_2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)
    
print("Data Loaded!!")


sss = ShuffleSplit(n_splits=1, test_size=0.3)
sss.get_n_splits(Xx, int_hosts)
train_index, test_index = next(sss.split(Xx, int_hosts)) 

train_data_tmp = Xx[train_index]
#start_time = time.time()
rbf_feature = RBFSampler(gamma=1, n_components=1000)
rbf_feature.fit(train_data_tmp)
Xxxx = rbf_feature.transform(Xx)

print("Data Tranformed with rows: ",len(Xxxx)," and columns: ",len(Xxxx[0]))


#####################
# https://adapt-python.github.io/adapt/generated/adapt.feature_based.WDGRL.html#adapt.feature_based.WDGRL.transform


# X = np.array(X_transformed)
X = np.array(Xxxx)
y = np.array(int_hosts)

total_splits = 1

sss = ShuffleSplit(n_splits=total_splits, test_size=0.9)

# for w in range(total_splits):

sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y)) 

Xs, Xt = X[train_index], X[test_index]
ys, yt = y[train_index], y[test_index]

#     Xs, ys, Xt, yt = make_classification_da()
model = WDGRL(lambda_=1., gamma=1., Xt=Xt, metrics=["acc"], random_state=0)
clf = model.fit(Xs, ys, epochs=10, verbose=0)
y_pred = clf.predict(Xt)
print(model.score(Xt, yt))
X_train = model.transform(Xs)
X_test = model.transform(Xt)

print("WDGRL Done!!")
######################

y_train = ys[:]
y_test = yt[:]


data_final = []
attribute_final = []
for i in range(len(y_train)):
    data_final.append(X_train[i])
    attribute_final.append(y_train[i])
    
for i in range(len(y_test)):
    data_final.append(X_test[i])
    attribute_final.append(y_test[i])
    


np.save(data_path + "WDGLE_Spike_seq_4380_Vector.npy",data_final)
np.save(data_path + "WDGLE_Spike_seq_4380_Attributes.npy",attribute_final)

print("All Processing Done!!!")
