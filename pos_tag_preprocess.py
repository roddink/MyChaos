# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:18:27 2017

@author: sz_rafahao
"""
import numpy as np

def data_loader(path,remove_empty = True):
    file = open(path)
#    file = open(path,encoding = 'utf8')
    text = file.readlines()
    for i in range(len(text)):
        text[i] = text[i].strip()
        text[i] = text[i].replace('[','')
        text[i] = text[i].replace(']nt','')
        text[i] = text[i].replace(']nz','')
        text[i] = text[i].replace(']ns','')
        text[i] = text[i].replace(']i','')
        text[i] = text[i].replace(']l','')

        text[i] = text[i].split(' ')
        for j in range(len(text[i])):

            temp = text[i][j].split('/')
            if len(temp)==2:# if correctly interpret, reserve the interpretation
                text[i][j] = temp
            elif len(temp)==1:# if lack of PoS label, give a dummy label for PoS label
                text[i][j] = [text[i][j],'CD']
            elif len(temp)>2:# if can't be interpreted correctly, keeps low level of label (can be wrong)
                print '/'.join(temp)
                
                text[i][j] = [temp[0],temp[1].replace(']','')]
            temp = []
            text[i][j][0] =text[i][j][0].decode('utf8')
            for k in range(len(text[i][j][0])): # Encode the words into "SBME" format
                if len(text[i][j][0])==1:
                    
                    temp.append([text[i][j][0][k].encode('utf8'),'S',text[i][j][1]])
                if len(text[i][j][0])==2:
                    if k == 0:
                        temp.append([text[i][j][0][k].encode('utf8'),'B',text[i][j][1]])
                    if k == 1:                    
                        temp.append([text[i][j][0][k].encode('utf8'),'E',text[i][j][1]])
                if len(text[i][j][0])>2:
                    if k == 0:
                        temp.append([text[i][j][0][k].encode('utf8'),'B',text[i][j][1]])
                    if k == len(text[i][j][0])-1:                    
                        temp.append([text[i][j][0][k].encode('utf8'),'E',text[i][j][1]])
                    if k < len(text[i][j][0])-1 and k>0:
                        temp.append([text[i][j][0][k].encode('utf8'),'M',text[i][j][1]])
            text[i][j] = temp
    temp_text = []
    for item in text:
        temp_text2 = []
        for item2 in item:
            for item3 in item2: 
                temp_text2.append(item3)
        if remove_empty == True: #Remove empty lines
            if temp_text2 != []:
                temp_text.append(temp_text2)
        else:
            temp_text.append(temp_text2)
    return temp_text

def get_item(data,index):
    text = []
    token_list = []
    for item in data:
        temp_text = []
        for item2 in item:
            temp_text.append(item2[index])
            token_list.append(item2[index])
        text.append(temp_text)
    return text,token_list

def get_joint(data): # making joint label such as 'B-NN'(beginning of a noun) 'E-NN' (endding of a noun) # not used
    text = []
    token_list = []
    for item in data:
        temp_text = []
        for item2 in item:
            temp_text.append(str(item2[1])+'-'+str(item2[2]))
            token_list.append(str(item2[1])+'-'+str(item2[2]))
        text.append(temp_text)
    return text,token_list

def get_length(data,max_len):# get a minimum value between length of the sentences and max_len
    length = []
    for item in data:
        length.append(min(len(item),max_len))
    return length

def Preprocess(path,shuffle=True,seed_path=None,tokenized = False):#main Func
    text = data_loader(path)# Loading data and label
    # 'text' is a nested list #sentences*sentences_length*[character,CWS label,PoS,label] 
    text_data,text_token = get_item(text,0)#get list of characters
    
    word_seg,word_seg_token = get_item(text,1) #get list of CWS labels
    pos,pos_token = get_item(text,2)# get list of POS labels
    joint,joint_token = get_joint(text)# get joint label
    length = get_length(text,500) # get length of each sentences
    if shuffle: # shuffle data, default to be True
        seed = np.random.permutation(len(length))
        if seed_path != None:
            seed = np.load(seed_path)
        np.save('seed.npy',seed)
        text_temp = text_data
        word_seg_temp = word_seg
        pos_temp = pos
        joint_temp = joint
        length_temp = length
        for ind in range(len(seed)):
            text_temp[ind] = np.array(text_data[seed[ind]])
            word_seg_temp[ind] = np.array(word_seg[seed[ind]])
            pos_temp[ind] = np.array(pos[seed[ind]])
            joint_temp[ind] = np.array(joint[seed[ind]])
            length_temp[ind] = np.array(length[seed[ind]])
        text_data = text_temp
        word_seg = word_seg_temp
        pos = pos_temp
        joint = joint_temp
        length = length_temp
    text_token = {k: v+1 for v, k in enumerate(list(set(text_token)))} # build dictionary for characters
    word_seg_token = {k: v for v, k in enumerate(list(set(word_seg_token)))} # build dictionary for CWS label
    pos_token = {k: v for v, k in enumerate(list(set(pos_token)))} # build dictionary for POS label
    joint_token = {k: v for v, k in enumerate(list(set(joint_token)))} # build dictionary for joint label
    max_len = 500
    if tokenized: # map the text to token, default to be False
        text_data = tokenizer(text_data,text_token,max_len)
        word_seg = tokenizer(word_seg,word_seg_token,max_len)
        pos = tokenizer(pos,pos_token,max_len)
        joint = tokenizer(joint,joint_token,max_len)
        print (max_len)
    return text_data,text_token,word_seg,word_seg_token,pos,pos_token,length,max_len,joint,joint_token

 
def tokenizer(text,token,max_len):# map text to token, INPUT: text, token dictionary, max_len
    text_new = []
    for i in range(len(text)):
        temp = np.zeros(max_len,dtype=int)
        for j in range(min(len(text[i]),max_len)):
            temp[j] = token.get(text[i][j],0)
        text_new.append(temp.tolist())
    return text_new
import pandas as pd    
def load_dictionary(path):#legacy
    file = open(path)
    dictionary = file.readlines()
    for ind in range(len(dictionary)):
        dictionary[ind] = dictionary[ind].strip()
    dictionary = pd.Series(dictionary)
    return dictionary

import re 
def check_in(dictionary,text,max_len):#legacy
    temp_vec = np.zeros((max_len))
    for m in re.finditer(dictionary,text):
        if m.end()-m.start() == 1:
            temp_vec[m.start()] = 1
        else:
            temp_vec[m.start():m.end()] = 2
            temp_vec[m.start()] = 1
            
        
    return temp_vec.tolist()
    
def check_in_dictionary(text,dictionary,max_len):#legacy
    dictionary = dictionary.apply(check_in,args=(text,max_len,))

    test = dictionary.values.tolist()
    result = np.max(np.array(test),axis=0)
    
    return result
    
def get_dict_feature(text,dict_path,max_len):#legacy
    dictionary = load_dictionary(dict_path)
#    file = open(path)
#    text = file.readlines()
    feature = []
    for item in text:
        feature.append(check_in_dictionary(item,dictionary,max_len))
    return np.array(feature)
    


def match(text,lm):#legacy
    result = lm.match_lexicon(text)
    result_vec = np.zeros((max(len(text.decode('utf8')),500)))
    for item in result:
        result_vec[item[0]:item[1]] = 1
    return result_vec[0:500]
        
def bigram_maker(tokenized_text):#get next character token, INPUT: Tokenized text
    bigram = []
    for item in tokenized_text:
        bigram.append([0]+item[:-1])
    return bigram

def trigram_maker(tokenized_text):#get previous character token, INPUT: Tokenized text
    bigram = []
    for item in tokenized_text:
        bigram.append(item[1:]+[0])
    return bigram

def uncut_letters(text,cut_result):#post processing
    letter_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','１','２','３','４','５','６','７','８','９','０']
    cut_result = np.array(cut_result)
    pos_list = [pos for pos, char in enumerate(text.decode('utf8')[:min(len(text.decode('utf8')),500)]) if char in letter_list]
    uncut_list = np.array(pos_list)
    if len(uncut_list)>0:
        cut_result[uncut_list] = 0
        uncut_list = uncut_list.tolist()
        for item in uncut_list:
            if (not((item-1) in uncut_list)):# check beginning of a non-Chinese substring
                cut_result[item] = 1
            if not((item+1) in uncut_list) and (item+1)<min(len(cut_result),500):#check end
                cut_result[item+1] = 1
    space_list = np.array([pos for pos, char in enumerate(text.decode('utf8')[:min(len(text.decode('utf8')),500)]) if char == ' '])#check spaces in the original text
    
    if len(space_list)>0:
        
        cut_result[space_list] = 0
    cut_result = cut_result.tolist()
    return cut_result

if __name__ == "__main__":
    text_data,text_token,word_seg,word_seg_token,pos,pos_token,length,num_words,joint,joint_token = Preprocess('data/CTB.txt',tokenized=False,shuffle=False)
    max_len = 500
    text_data = tokenizer(text_data,text_token,max_len)
    text_data_bi = bigram_maker(text_data)
    text_data_tri = trigram_maker(text_data)
    word_seg = tokenizer(word_seg,word_seg_token,max_len)
    pos = tokenizer(pos,pos_token,max_len)    
