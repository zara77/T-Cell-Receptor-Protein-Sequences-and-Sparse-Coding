#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import timeit
from itertools import product
import timeit
import math
import pandas as pd

from itertools import product
import timeit
import math


# In[3]:
#location_id_val = 1559
#location_id_val = 480
#location_id_val = 18728
#location_id_val = 1041
#location_id_val = 944
#location_id_val = 14248

#unique_char_1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzαßΓπΣσµτΦΘΩδ∞φε∩±≥≤⌠⌡≈√€†‡‰ŠŒŽšœžŸ¤¦©¼½¾ÀÁÂÃÄÅÆÇÈÉÊËÌÍÏÏÐÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëïðòóôõö÷øúûüùýþÿ9'
unique_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers



#data_path = "/olga-data0/Sarwan/Adversarial_attack/Feature_vectors/"
#data_name = "org_red_seq_8220"

data_path = "/alina-data1/Zara/TCell_Receptor/Dataset/"
data_name = "splitted_sequences_2"

seq_data = np.loadtxt(data_path+ data_name + ".npy",dtype=str)

# gmer_length = 7 # 9 , 5
spaced_kmer_length = 3 # 6 , 4


Kmer = spaced_kmer_length

#unique_char = unique_char_1[0:location_id_unq[qww]] + "9"
unique_seq_kmers_final_list = [''.join(c) for c in product(unique_char, repeat=spaced_kmer_length)]  


start = timeit.default_timer()

final_feature_vector = []

for seq_ind in range(len(seq_data)):
    print("index: ",seq_ind,"/",len(seq_data))
    se_temp = seq_data[seq_ind]


    k_mers_final = build_kmers(se_temp,spaced_kmer_length)


#     #extract spaced kmers
#     spaced_kmers = []
#     for i in range(len(gmers_list)):
#         temp_val = gmers_list[i]
#         spaced_kmers.append(temp_val[0:spaced_kmer_length])

#     k_mers_final = spaced_kmers[:]
    ################ Generate PWM (Start) #########################

#     14248,1559,480,18728,1041,944
#     138,30,19,135,26,18
    #character_val = location_id_unq[qww]
    character_val = len(unique_char)
    pwm_matrix = np.array([[0]*Kmer]*(character_val))

#     a_val = [0]*Kmer

    # input_file = open("E:/RA/Position Weight Matrix/Code/EI_nine.txt","r")   
    count_lines = 0 # Initialize the total number of sequences to 0
    # Read line by line, stripping the end of line character and
    # updating the PWM with the frequencies of each base at the 9 positions
    for ii in range(len(k_mers_final)):
        line = k_mers_final[ii]
        count_lines += 1 # Keep counting the sequences
        for i in range(len(line)):
            if line[i]=='9':
                pwm_matrix[len(pwm_matrix)-1,i] = pwm_matrix[len(pwm_matrix)-1,i] + 1
            else:
                ind_tmp = unique_char.index(line[i])
                pwm_matrix[ind_tmp,i] = pwm_matrix[ind_tmp,i] + 1


    LaPlace_pseudocount = 0.1
    equal_prob_nucleotide = character_val/100

    for i in range(len(k_mers_final[0])):

        for x in range(len(pwm_matrix)):
            pwm_matrix[x,i] = round(math.log((pwm_matrix[x,i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)


    ################ Generate PWM (End) #########################

    ################ Assign Individual k-mers Score (Start) #########################
    each_k_mer_score = []
    listofzeros = [0] * len(unique_seq_kmers_final_list)
    for ii in range(len(k_mers_final)):
        line = k_mers_final[ii]
        score = 0
        for i in range(len(line)):
            if line[i]=='9':
                score += pwm_matrix[len(pwm_matrix)-1,i]
            else:
                ind_tmp = unique_char.index(line[i])
                score += pwm_matrix[ind_tmp,i]



        final_score_tmp = round(score, 3)
        each_k_mer_score.append(final_score_tmp)

        ###################### assign weughted k-mers frequency score ###############
        kmer_val_check = str(line)
        aa_lst_1 = kmer_val_check.replace(",","")
        aa_lst_2 = aa_lst_1.replace("[","")
        aa_lst_3 = aa_lst_2.replace("\"","")
        aa_lst_4 = aa_lst_3.replace("]","")
        aa_lst_5 = aa_lst_4.replace("'","")
        aa_lst_6 = aa_lst_5.replace(" ","")


        ind_tmp = unique_seq_kmers_final_list.index(aa_lst_6)
        listofzeros[ind_tmp] = listofzeros[ind_tmp] + (1 * final_score_tmp)


#     final_feature_vector.append(each_k_mer_score)
    final_feature_vector.append(listofzeros)
    ################ Assign Individual k-mers Score (end) #########################

stop = timeit.default_timer()
print("PWM Time : ", stop - start)

max_vec_length = 0

for t in range(len(final_feature_vector)):
    if len(final_feature_vector[t]) > max_vec_length:
        max_vec_length = len(final_feature_vector[t])

padded_pwm_vec = []

for t in range(len(final_feature_vector)):
    row_vec = final_feature_vector[t]
    if(len(row_vec)<max_vec_length):
        for k in range(len(row_vec),max_vec_length):
            row_vec.append(0)
    padded_pwm_vec.append(row_vec)
np.save(data_path + "PWM2Vec_Padded_" + data_name + ".npy",padded_pwm_vec)


print("All Processing Done!!!")




