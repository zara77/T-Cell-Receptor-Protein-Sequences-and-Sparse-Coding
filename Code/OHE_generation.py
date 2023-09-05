import pandas as pd
import numpy as np





data_path = "/alina-data1/Zara/TCell_Receptor/Dataset/"
data_name = "splitted_sequences_2"

seq_data = np.loadtxt(data_path+ data_name + ".npy",dtype=str)

print("Data Loaded!!!")

pattern_holder = seq_data[:]

max_sequence_length = 0
min_sequence_length = 100
avg_sequence_length = 0

for i in range(len(pattern_holder)):
    avg_sequence_length = avg_sequence_length + len(pattern_holder[i])
    if len(pattern_holder[i])>=max_sequence_length:
        max_sequence_length = len(pattern_holder[i])
    if len(pattern_holder[i])<=min_sequence_length:
        min_sequence_length = len(pattern_holder[i])

# Getting the unique values
res = list(set(i for j in pattern_holder for i in j))

# printing result
print ("Unique values : ", str(res))
print ("Unique values Length : ", len(res))
print ("Max Sequence Length : ", max_sequence_length)
print ("Min Sequence Length : ", min_sequence_length)
print ("Average Sequence Length : ", avg_sequence_length/len(pattern_holder))

# np.array(res).to_categorical() 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(res)
integer_encoded,max(integer_encoded)

final_data = []
for ind in range(len(pattern_holder)):
    asd = pattern_holder[ind]
    for i in range(len(res)):
        #asd = np.where(asd == res[i], integer_encoded[i], asd)
        asd = [integer_encoded[i] if item == res[i] else item for item in asd]
    final_data.append(asd)

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

one_hot_data = []
ohe_vector_length = len(res)*max_sequence_length
for i in range(len(final_data)):
    if i%100==0:
        print("OHE index: ",i,"/",len(final_data))
    row_wise = final_data[i]
    row_vector = []
    for j in range(len(row_wise)):
        temp_vector = [0]* len(res)
        temp_vector[int(row_wise[j])] = 1
        row_vector.append(list(temp_vector))
    row_vec = flatten_list(row_vector)
    if(len(row_vec)<ohe_vector_length):
        for k in range(len(row_vec),ohe_vector_length):
            row_vec.append(0)
    one_hot_data.append(row_vec)

np.save(data_path + "OHE_" + data_name + ".npy",one_hot_data)

print("All processing done!!")

