import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure that dna sequences and labels correspond to each other.
# dna_sequences labels
# site          class
# 200-SS-200    label
def sequence_concat_label(seq_path,label):
    splice_site_sequence = pd.read_csv(seq_path,sep='\t',header=None)
    splice_site_sequence = pd.DataFrame(splice_site_sequence)
    splice_site_sequence.columns = ['site']
    sequence_label_dataframe = splice_site_sequence.reindex(columns=['site','class'],fill_value=label)
    
    return sequence_label_dataframe

# ['A','T','G','C','N']->['1000','0100','0010','0001','0000']
# ['site','non_site']->['10','01']
def one_hot_encode(sequence_label_dataframe):
    sequence_label_dataframe['site'] = sequence_label_dataframe['site'].str.replace('A','1000')
    sequence_label_dataframe['site'] = sequence_label_dataframe['site'].str.replace('T','0100')
    sequence_label_dataframe['site'] = sequence_label_dataframe['site'].str.replace('G','0010')
    sequence_label_dataframe['site'] = sequence_label_dataframe['site'].str.replace('C','0001')
    sequence_label_dataframe['site'] = sequence_label_dataframe['site'].str.replace('[A-Z]','0000',regex=True)
    sequence_label_dataframe['class'] = sequence_label_dataframe['class'].str.replace('non_site','01')
    sequence_label_dataframe['class'] = sequence_label_dataframe['class'].str.replace('site','10')

    return sequence_label_dataframe

# reshape input -> (-1,402,4)
# reshape label -> (-1,2)
def reshape_one_hot_data(one_hot_site_data,ono_hot_non_site_data):
    data = pd.concat([one_hot_site_data,ono_hot_non_site_data],ignore_index=True)
    x = []
    y = []

    x = np.array(data['site'].values, dtype=str)
    x = x.view('U1').reshape(x.shape+(-1,))
    x_width = 4
    x_height = int(x.shape[1]/x_width)
    x = x.reshape(-1,x_height,x_width)

    y = np.array(data['class'].values, dtype=str)
    y = y.view('U1').reshape(y.shape+(-1,))
    y = y.reshape(-1,2)

    x_array = np.array(x,dtype=float)
    y_array = np.array(y,dtype=float)

    return x_array,y_array

def encode_and_split_data(splice_site_seq_path,non_splice_site_seq_path):
    # -----------Organize splice site sequences-----------
    site_seq_label = sequence_concat_label(splice_site_seq_path,'site')
    # -----------Organize non splice site sequences-----------
    non_site_seq_label = sequence_concat_label(non_splice_site_seq_path,'non_site')
    # -----------One-hot coding of splice site sequences-----------
    one_hot_site_data = one_hot_encode(site_seq_label)
    # -----------One-hot coding of non splice site sequences-----------
    one_hot_non_site_data = one_hot_encode(non_site_seq_label)
    # -----------Converting shapes and formats for one-hot coded datasets-----------
    x,y = reshape_one_hot_data(one_hot_site_data,one_hot_non_site_data)

    # train_dataset:val_dataset:test_dataset=0.60:0.15:0.25
    x_train_val,x_test,y_train_val,y_test = train_test_split(x,y,test_size=0.25,random_state=10,shuffle=True)
    x_train,x_val,y_train,y_val = train_test_split(x_train_val,y_train_val,test_size=0.20,random_state=10,shuffle=True)

    return x_train,x_val,x_test,y_train,y_val,y_test
