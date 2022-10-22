import numpy as np
import os
from sklearn.model_selection import train_test_split

def names_list(save_path):
    '''
    For each file name, this function returns a list [action, sample_number, filename]. This helps us load a directory such that 
    dict={action: {sample: numpy file}}
    Args:
    save_path: path to load the numpy files from
    Returns:
    A list where each element is another list = [action, sample_number, filename]
    '''
    file_params=[]
    for filename in os.listdir(save_path):
        if filename.endswith('.npy'):
            file_params.append((filename.split('_')[0],filename.split('_')[1],filename))
    print('Total file count:',len(file_params))
    return file_params 

def create_final_features(save_path, file_params):
    '''
    Creates the final feature numpy array from the dump of all arrays - one for each frame. 
    Requires the filename of each frame's numpy array to be in the format 'action_samplenumber_framenumber'
     Agruments:
    Path for loading the numpy arrays of each frame
    A list where each element is another list = [action, sample_number, filename]
     Returns:
     Numpy array of shape samples X token_no. X token length
    '''
    labels=[]
    act_array=[]
    dict_act={}
    for (a,s,f) in file_params:
        npy_arr=np.load(os.path.join(save_path,f))
        if a in dict_act:
            if s in dict_act[a]:
                dict_act[a][s].append(npy_arr)
            else:
                dict_act[a][s]=[npy_arr]
        else:
            dict_act[a]={s:[npy_arr]}
    for a in dict_act:
        samp_array=[]
        for s in dict_act[a]:
            samp_array.append(np.array(dict_act[a][s]).squeeze())
            labels.append(a)
        act_array.append(np.array(samp_array))
    final_array=np.vstack(act_array)
    return final_array, labels

def create_labels_dict(labels_list):
    '''
    Create dict from label for forward and reverse lookup
    args: List of labels
    returns:
    dict_labels = {0:'label1',1:'label2'....}
    dict_reverse_label={'label1':0, 'label2':0......}
    
    '''
    dict_labels={}
    dict_reverse_labels={}
    for i,l in enumerate(list(set(labels_list))):
        dict_labels[i]=l
    for k,v in dict_labels.items():
        dict_reverse_labels[v]=k
    print(dict_labels)
    print(dict_reverse_labels)
    return dict_labels, dict_reverse_labels

def OHK(list1,dict1): #Create the one hot key encoding of the labels
    label_idx=[dict1[l] for l in list1]
    ohk_labels=np.zeros((len(list1),len(set(list1))))
    ohk_labels[range(len(list1)),label_idx]=1
    return ohk_labels

def train_test_extract(features_array, labels_array,train_split=0.95,shuffle_count=1000):
    '''
    Consolidates the features and one-hot-encoded labels, shuffles the dataset and splits them into train and test arrays
    
    '''
    batch_size=features_array.shape[0]
    sample_size=features_array.shape[1]
    feature_size=features_array.shape[2]
    new_features_array=features_array.reshape(batch_size,-1)
    print(new_features_array.shape)
    print(labels_array.shape)
    comb_numpy = np.hstack((new_features_array,labels_array))
    for _ in range(shuffle_count):
        np.random.shuffle(comb_numpy)
    X_train, X_test, y_train, y_test =train_test_split(features_array,labels_array,train_size=train_split,shuffle=True, stratify=labels_array)
    X_train=X_train.reshape(-1,sample_size, feature_size)
    X_test=X_test.reshape(-1,sample_size, feature_size)
    return X_train, X_test,y_train, y_test