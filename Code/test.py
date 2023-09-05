#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install libsvm


# In[2]:


import os

# Specify the new directory path
new_directory = 'E:/RA/DNA_Binding_Protein/Code/MLapSVM-LBS-main/MLapSVM-LBS-main/'

# Change the current working directory to the new directory
os.chdir(new_directory)

# You can verify the change by printing the current working directory
print("Current working directory:", os.getcwd())


# In[3]:


import numpy as np
import sklearn
from sklearn.metrics.pairwise import rbf_kernel
from utils.alg import Alg
from utils.cons import Cons
from sklearn import preprocessing
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import utils.tools as utils
from sklearn.preprocessing import StandardScaler


# # LapSVM

# In[4]:


class LapSVM():
    def __init__(self, gamma=0.1, c=0.03125, lamda2=0.7,lamda1=0.9, k=8):
        self.gamma = gamma  # 核矩阵gamma
        self.alg = Alg(c, lamda1, lamda2)
        self.cons = Cons(k)
    
    #单图
    def train(self,traindata, testdata, trainlabel):
        l = traindata.shape[0]
        u = testdata.shape[0]
        testlabel = np.zeros((u, 1))
        label = np.concatenate((trainlabel, testlabel), axis=0)
        data = np.concatenate((traindata, testdata), axis=0)
        K = rbf_kernel(data, data, self.gamma)
        L=self.cons.construct_LBS(data,label)
        score=self.alg.solve_alpha(L,K,trainlabel)
        pred_score = score[l:]
        return pred_score
    
class LapSVM_M():
    def __init__(self, gamma, c=0.03125,  lamda2=1,lamda1=1, k=6):
        self.gamma = gamma  # 核矩阵gamma
        self.alg = Alg(c, lamda1, lamda2)
        self.cons = Cons(k)

    def train(self, traindata, testdata, trainlabel):
        l = traindata.shape[0]
        u = testdata.shape[0]
        testlabel = np.zeros((u, 1))
        label = np.concatenate((trainlabel, testlabel), axis=0)
        data = np.concatenate((traindata, testdata), axis=0)
        K = rbf_kernel(data, data, self.gamma)
        
#         print("label: ",label)

        L1 = self.cons.intergrate(data, label)
        score = self.alg.train(K, L1, trainlabel)
        pred_score = score[l:]
        # pred_score[np.where(pred_score>=0)]=1
        # pred_score[np.where(pred_score<0)]=-1
        return pred_score

def loaddata(filename1,filename2):
    train_data=sio.loadmat(filename1)
    test_data=sio.loadmat(filename2)
    GE_train=train_data['GE_1075']
    NMBAC_train=train_data['NMBAC_1075']
    Pse_train=train_data['PSSM_Pse_1075']
    trainlabel=train_data['label_1075']
    traindata = np.concatenate((GE_train,NMBAC_train,Pse_train),axis=1)
    
    GE_test=test_data['GE_186']
    NMBAC_test=test_data['NMBAC_186']
    Pse_test=test_data['PSSM_Pse_186']
    testlabel=test_data['label_186']
    testdata = np.concatenate((GE_test,NMBAC_test,Pse_test),axis=1)
    
    data =np.concatenate((traindata,testdata),axis=0)
    min_max_scaler=preprocessing.MinMaxScaler()
    data=min_max_scaler.fit_transform(data)
    traindata=data[:1075,:]
    testdata=data[1075:,:]
    traindata,trainlabel = sklearn.utils.shuffle(traindata, trainlabel,random_state=1)
    testdata,testlabel =sklearn.utils.shuffle(testdata,testlabel,random_state=1)
    traindata,x=np.unique(traindata,axis=0,return_index=True)
    ls=trainlabel[x]
    return traindata,ls,testdata,testlabel




# In[5]:


# # if __name__ == '__main__':
# lapsvm = LapSVM(gamma=0.125,c=8,lamda2=1,lamda1=0.3)

# traindata,trainlabel,testdata,testlabel=loaddata("dataset/PDB1075_feature.mat","dataset/PDB186_feature.mat")

# predlabel=lapsvm.train(traindata,testdata,trainlabel)
# auc=roc_auc_score(testlabel,predlabel)
# predlabel[np.where(predlabel>=0)]=1
# predlabel[np.where(predlabel<0)]=0
# testlabel[np.where(testlabel<0)]=0
# tn, fp, fn, tp = confusion_matrix(testlabel,predlabel).ravel()
# SN=tp/(tp+fn)
# SP=tn/(tn+fp)
# acc=(tp+tn)/(tp+fn+tn+fp)
# mcc=matthews_corrcoef(testlabel,predlabel)
# print(acc)


# In[6]:


data_14189 = sio.loadmat("dataset/Data_sets_PDB14189_2272.mat")

GE_14189 =data_14189['GE_14189']
NMBAC_14189 =data_14189['NMBAC_14189']
MCD_14189 =data_14189['MCD_14189']
Pse_14189 =data_14189['PSSM_Pse_14189']
testlabel_14189=data_14189['label_14189']


GE_2272 =data_14189['GE_2272']
NMBAC_2272 =data_14189['NMBAC_2272']
MCD_2272 =data_14189['MCD_2272']
Pse_2272 =data_14189['PSSM_Pse_2272']
testlabel_2272=data_14189['label_2272']


# In[7]:


data_1075 = sio.loadmat("dataset/PDB1075_feature.mat")

GE_1075 =data_1075['GE_1075']
NMBAC_1075 =data_1075['NMBAC_1075']
MCD_1075 =data_1075['MCD_1075']
Pse_1075 =data_1075['PSSM_Pse_1075']
testlabel_1075 =data_1075['label_1075']


# In[8]:


data_186 = sio.loadmat("dataset/PDB186_feature.mat")

GE_186 =data_186['GE_186']
NMBAC_186 =data_186['NMBAC_186']
MCD_186 =data_186['MCD_186']
Pse_186 =data_186['PSSM_Pse_186']
testlabel_186 =data_186['label_186']


# In[ ]:





# In[9]:


# dataset = GE_186[:]
# dataset = NMBAC_186[:]
# dataset = MCD_186[:]
# dataset = Pse_186[:]
# dataset = np.hstack((GE_186[:], NMBAC_186[:], MCD_186[:], Pse_186[:]))

# label = testlabel_186[:]


# dataset = GE_1075[:]
# dataset = NMBAC_1075[:]
# dataset = MCD_1075[:]
# dataset = Pse_1075[:]
# dataset = np.hstack((GE_1075[:], NMBAC_1075[:], MCD_1075[:], Pse_1075[:]))

# label = testlabel_1075[:]

# dataset = GE_2272[:]
# dataset = NMBAC_2272[:]
# dataset = MCD_2272[:]
# dataset = Pse_2272[:]
# dataset = np.hstack((GE_2272[:], NMBAC_2272[:], MCD_2272[:], Pse_2272[:]))

# label = testlabel_2272[:]

dataset = GE_14189[:]
# dataset = NMBAC_14189[:]
# dataset = MCD_14189[:]
# dataset = Pse_14189[:]
# dataset = np.hstack((GE_14189[:], NMBAC_14189[:], MCD_14189[:], Pse_14189[:]))

label = testlabel_14189[:]

dataset.shape


# In[10]:


path = "result/DNA/"
if not os.path.exists(path):
    os.makedirs(path)

# #dataset_name = "PDB14189"
# dataset_name = "PDB2272"
# #dataset_name = "PDB1075"
dataset_name = "PDB186"

start = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
fw_perf = open(path + dataset_name + "_DNA_binding" + str(start).replace(":","_") + ".txt", 'w')


cross_folds = 5 # 5
# total_epochs = 10 # 50


scores = []
k_fold = cross_folds
# top_words = 20
# max_review_length = 800
# embedding_vecor_length = 128



for index in range(1):
    
    
#     # Reshape to 2D
#     dataset = dataset_tmp.reshape(len(dataset_tmp), -1)
#     #dataset = dataset.tolist()
#     label = np.array(label)
    
    # Rest of your code ...

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1024)
    for ((train_index, test_index), k) in zip(skf.split(dataset, label), range(k_fold)):
        #encoded_docs_train = [one_hot(d, top_words) for d in np.array(dataset)[train_indices]]
        #encoded_docs_test = [one_hot(d, top_words) for d in np.array(dataset)[test_indices]]

        #X_train = sequence.pad_sequences(encoded_docs_train, maxlen=max_review_length, padding='pre', value=0.0)
        #X_test = sequence.pad_sequences(encoded_docs_test, maxlen=max_review_length, padding='pre', value=0.0)
        
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = label[train_index], label[test_index]


        ################################################
#         lapsvm = LapSVM(gamma=0.125,c=8,lamda2=1,lamda1=0.3)
        lapsvm = LapSVM_M(gamma=0.125,c=8,lamda2=1,lamda1=0.3)
        
        predictions=lapsvm.train(X_train,X_test,y_train)
        ################################################

        # prediction probability
#         y_test = utils.to_categorical(label[test_index])
#         predictions = model.predict(X_test)

#         auc=roc_auc_score(testlabel,predlabel)
    
        predictions_prob = predictions[:]
        auc_ = roc_auc_score(label[test_index], predictions_prob)
        pr = average_precision_score(label[test_index], predictions_prob)

#         y_class = utils.categorical_probas_to_classes(predictions)
#         # true_y_C_C=utils.categorical_probas_to_classes(true_y_C)
#         true_y = utils.categorical_probas_to_classes(y_test)
        
        ##############################
        y_class = predictions[:]
        true_y = y_test[:]
        
        y_class[np.where(predictions>=0)]=1
        y_class[np.where(predictions<0)]=-1
#         true_y[np.where(predictions<0)]=0
        ##############################
        
        acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(
            len(y_class), y_class, true_y)
        print("======================")
        print("======================")
        print("Iter " + str(index) + ", " + str(k + 1) + " of " + str(k_fold) +
              "cv:")
        print(
            '\tacc=%0.4f,pre=%0.4f,npv=%0.4f,sn=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
            % (acc, precision, npv, sensitivity, specificity, mcc, f1))
        print('\tauc=%0.4f,pr=%0.4f' % (auc_, pr))

        fw_perf.write(
            str(acc) + ',' + str(precision) + ',' + str(npv) + ',' +
            str(sensitivity) + ',' + str(specificity) + ',' + str(mcc) +
            ',' + str(f1) + ',' + str(auc_) + ',' + str(pr) + '\n')

        scores.append([
            acc, precision, npv, sensitivity, specificity, mcc, f1, auc_,
            pr
        ])

scores = np.array(scores)
print(len(scores))
print("acc=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100))
print("precision=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100))
print("npv=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100))
print("sensitivity=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100))
print("specificity=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100))
print("mcc=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100))
print("f1=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100))
print("roc_auc=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100))
print("roc_pr=%.2f%% (+/- %.2f%%)" %
      (np.mean(scores, axis=0)[8] * 100, np.std(scores, axis=0)[8] * 100))

end = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print('start: %s' % start)
print('end: %s' % end)

fw_perf.write(
    "acc=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100) +
    '\n')
fw_perf.write(
    "precision=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100) +
    '\n')
fw_perf.write(
    "npv=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100) +
    '\n')
fw_perf.write(
    "sensitivity=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100) +
    '\n')
fw_perf.write(
    "specificity=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100) +
    '\n')
fw_perf.write(
    "mcc=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100) +
    '\n')
fw_perf.write(
    "f1=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100) +
    '\n')
fw_perf.write(
    "roc_auc=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100) +
    '\n')
fw_perf.write(
    "roc_pr=%.2f%% (+/- %.2f%%)" %
    (np.mean(scores, axis=0)[8] * 100, np.std(scores, axis=0)[8] * 100) +
    '\n')
fw_perf.write('start: %s' % start + '\n')
fw_perf.write('end: %s' % end)

fw_perf.close()


# In[ ]:


print("Done")


# In[ ]:





# In[ ]:





# In[ ]:




