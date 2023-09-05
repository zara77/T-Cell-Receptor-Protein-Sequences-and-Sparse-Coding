#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import timeit
from itertools import product
import timeit
import math
import pandas as pd

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers
    
    
#unique_char_1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzαßΓπΣσµτΦΘΩδ∞φε∩±≥≤⌠⌡≈√€†‡‰ŠŒŽšœžŸ¤¦©¼½¾ÀÁÂÃÄÅÆÇÈÉÊËÌÍÏÏÐÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëïðòóôõö÷øúûüùýþÿ9'
unique_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'





# for location_id_ind in location_id_val:

data_path = "/alina-data1/Zara/TCell_Receptor/Dataset/"
data_name = "splitted_sequences_2"

seq_data = np.loadtxt(data_path+ data_name + ".npy",dtype=str)




unique_in_data = list(set(i for j in seq_data for i in j))


seq_val_global = []
for u in range(len(seq_data)):
    tmp_seq_val = seq_data[u]

    val_tmp = str(tmp_seq_val)
    val_tmp_1 = val_tmp.replace(", ","")
    val_tmp_2 = val_tmp_1.replace("]","")
    val_tmp_3 = val_tmp_2.replace("[","")
    val_tmp_4 = val_tmp_3.replace("\'","")
    seq_val_global.append(val_tmp_4)



# In[11]:


print("Total Rows: ",len(seq_val_global),", Vector Length: ",len(seq_val_global[100]))







# In[18]:




# gmer_length = 7 # 9 , 5
spaced_kmer_length = 3 # 6 , 4

Kmer = spaced_kmer_length



unique_seq_kmers_final_list = [''.join(c) for c in product(unique_char, repeat=spaced_kmer_length)]  


start = timeit.default_timer()

frequency_vector = []

for seq_ind in range(len(seq_val_global)):
    if seq_ind%1000==0:
        print("nGram index: ",seq_ind,"/",len(seq_val_global))
    se_temp = seq_val_global[seq_ind]


    k_mers_final = build_kmers(se_temp,spaced_kmer_length)

    #create dictionary
    idx = pd.Index(k_mers_final) # creates an index which allows counting the entries easily
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


# In[19]:


np.save(data_path + "Spike2Vec_" + data_name + ".npy",frequency_vector)

print("Embedding Saved!!")



print("All Processing Done!!!")




