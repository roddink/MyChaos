# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:33:29 2017
python 2.7, tensorflow 1.0, gensim
@author: sz_rafahao
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"#GPU setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from langconv import *
def init_embedding(gensim_model,embedding_size): # tra2sim embedding initialization
    tra_keys = gensim_model.wv.vocab.keys()
    sim_keys = []
    for item in tra_keys:
        sim_keys.append(Converter('zh-hans').convert(item))
#    sim_keys = list(set(sim_keys))
    tra2sim_dict = dict(zip(sim_keys,tra_keys))
    dicts = list(enumerate(list(set(sim_keys))))
    print dicts[0]
    dicts = map (lambda t: (t[1].encode('utf8'), t[0]+1), dicts)
    dicts =dict(dicts)
    embedding = np.zeros((len(dicts)+1,embedding_size))
    
    for item in tra2sim_dict.keys():
        embedding[dicts[item.encode('utf8')],:] = gensim_model[tra2sim_dict[item]]/(sum(gensim_model[tra2sim_dict[item]]*gensim_model[tra2sim_dict[item]])**0.5)
        
    return embedding,dicts,tra2sim_dict


#batch makers
def batch_make(text_data,word_seg,pos,length,batch_size):#legacy
    seed = np.random.permutation(len(text_data))
    text_data_temp = text_data
    word_seg_temp = word_seg
    pos_temp = pos
    length_temp = length
    for ind in range(len(text_data)):
        text_data[ind] = text_data_temp[seed[ind]]
        word_seg[ind] = word_seg_temp[seed[ind]]
        pos[ind] = pos_temp[seed[ind]]
        length[ind] = length_temp[seed[ind]]
    batch = []
    for ind in range(0,len(text_data),batch_size):
        batch.append([text_data[ind:ind+batch_size],word_seg[ind:ind+batch_size],pos[ind:ind+batch_size],length[ind:ind+batch_size]])
    return batch[:-1]
    
def batch_make_joint(text_data,joint,length,batch_size):#legacy
    seed = np.random.permutation(len(text_data))
    text_data_temp = text_data
    joint_temp = joint
    length_temp = length
    for ind in range(len(text_data)):
        text_data[ind] = text_data_temp[seed[ind]]
        joint[ind] = joint_temp[seed[ind]]
        length[ind] = length_temp[seed[ind]]
    batch = []
    for ind in range(0,len(text_data),batch_size):
        batch.append([text_data[ind:ind+batch_size],joint[ind:ind+batch_size],length[ind:ind+batch_size]])
    return batch

def batch_make_dict(text_data,text_data_bi,text_data_tri,word_seg,pos,dictionary,length,batch_size):
    batch = []
    for ind in range(0,len(text_data),batch_size):
        batch.append([text_data[ind:ind+batch_size],text_data_bi[ind:ind+batch_size],text_data_tri[ind:ind+batch_size],word_seg[ind:ind+batch_size],pos[ind:ind+batch_size],dictionary[ind:ind+batch_size],length[ind:ind+batch_size]])
    return batch[:-1]



from pos_tag_preprocess import *
from lexicon_features_cws import *
def file_2_dict(path):# input .npy file 
    token_file = np.load(path)
    keys = []
    values = []
    for item,item2 in zip(token_file[0],token_file[1]):
        keys.append(item.decode('utf8').encode('utf8'))
        values.append(int(item2))
    return dict(zip(keys,values))
# parameter settings.
num_examples = 100
num_features = 200
n_hidden= 100
embedding_size =300
gensim_model = Word2Vec.load('data/mongoDB_w2v_epoch5.txt')# init_embedding 
max_len = 500
n_dict_feature = 3
#Preprocess Begins
print 'Pre-process begins ...'
print 'print the words cannot be interpreted correctly ...'
text_data,text_token,word_seg,word_seg_token,pos,pos_token,length,num_words,joint,joint_token = Preprocess('data/CTB.txt',tokenized=False,shuffle=False)
text_data2,text_token2,word_seg2,word_seg_token2,pos2,pos_token2,length2,num_words2,joint2,joint_token2 = Preprocess('data/new_pku_2014.txt',tokenized=False,shuffle=False)
text_data3,text_token3,word_seg3,word_seg_token3,pos3,pos_token3,length3,num_words3,joint3,joint_token3 = Preprocess('data/ltp_standard.txt',tokenized=False,shuffle=False)
#
print 'Prepare data for dictionary_feature extraction ....'
text_data_new = text_data+text_data2
text_data_text = []
text_data2_text = []
text_data3_text = []


print "load presaved label dictionary:"
word_seg_token2 = file_2_dict('CWS_POS_cws2_embedding.npy')
pos_token2 = file_2_dict('CWS_POS_pos2_embedding.npy')

for ind in range(len(text_data)):
    text_data_text.append(''.join(text_data[ind]))
for ind in range(len(text_data2)):
    text_data2_text.append(''.join(text_data2[ind]))
for ind in range(len(text_data3)):
    text_data3_text.append(''.join(text_data3[ind]))


print 'dictionary feature extraction begins:'
#dictionary features extraction
lm = lexicon_feat('data/dictionary2.txt')

text_data_dict = []


for item in text_data_text:
    text_data_dict.append(lm.produce_features(item,max_len=max_len))

text_data2_dict = []
for item in text_data2_text:
    text_data2_dict.append(lm.produce_features(item,max_len=max_len))

text_data3_dict = []
for item in text_data3_text:
    text_data3_dict.append(lm.produce_features(item,max_len=max_len))

del text_data_text
del text_data2_text
del text_data3_text
print 'initialize pretrained embedding ...'
pretrain_embed,text_token,_= init_embedding(gensim_model,embedding_size)
#Tokenize data
print 'map text and labels to tokens'
print 'Print a sample of text, tokenized text, bigramed_token and its length (for debug purpose)....'

text_data = tokenizer(text_data,text_token,max_len)
text_data_bi = bigram_maker(text_data)
text_data_tri = trigram_maker(text_data)
word_seg = tokenizer(word_seg,word_seg_token,max_len)
pos = tokenizer(pos,pos_token,max_len)


print text_data2[0]
text_data2 = tokenizer(text_data2,text_token,max_len)
text_data2_bi = bigram_maker(text_data2)
text_data2_tri = trigram_maker(text_data2)

print text_data2[0]
print text_data2_bi[0]
print len(text_data2[0])

word_seg2 = tokenizer(word_seg2,word_seg_token2,max_len)
pos2 = tokenizer(pos2,pos_token2,max_len)

text_data3 = tokenizer(text_data3,text_token,max_len)
text_data3_bi = bigram_maker(text_data3)
text_data3_tri = trigram_maker(text_data3)

print text_data3[0]
print text_data3_bi[0]
print len(text_data3[0])

word_seg3 = tokenizer(word_seg3,word_seg_token2,max_len)
pos3 = tokenizer(pos3,pos_token2,max_len)


batch_size = 100
num_examples = batch_size
num_class_cws = len(word_seg_token)
num_class_pos = len(pos_token)
num_class_pos_2 = len(pos_token2)
num_class_pos_3 = 20#needs to be specified
print (word_seg_token)
print (word_seg_token2)




# sort and shuffle data 
print 'Sort data by its sequence length:'
seed = np.random.permutation(len(text_data))
seed[:3000].sort()
seed[3000:].sort()
temp_data = []
temp_data_bi = []
temp_data_tri = []
temp_seg = []
temp_pos = []
temp_length = []
temp_data_dict = []
for item in seed:
    temp_data.append(text_data[item])
    temp_data_bi.append(text_data_bi[item])
    temp_data_tri.append(text_data_tri[item])
    temp_seg.append(word_seg[item])
    temp_pos.append(pos[item])
    temp_length.append(length[item])
    temp_data_dict.append(text_data_dict[item])
text_data = temp_data
text_data_bi = temp_data_bi
text_data_tri = temp_data_tri
word_seg = temp_seg
pos = temp_pos
text_data_dict = temp_data_dict
length = temp_length
seed = np.random.permutation(len(text_data2))
seed[:10000].sort()
seed[10000:].sort()
temp_data = []
temp_data_bi = []
temp_data_tri = []
temp_seg = []
temp_pos = []
temp_length = []
temp_data_dict = []
for item in seed:
    temp_data.append(text_data2[item])
    temp_data_bi.append(text_data2_bi[item])
    temp_data_tri.append(text_data2_tri[item])
    temp_seg.append(word_seg2[item])
    temp_pos.append(pos2[item])
    temp_length.append(length2[item])
    temp_data_dict.append(text_data2_dict[item])
text_data2 = temp_data
text_data2_bi = temp_data_bi
text_data2_tri = temp_data_tri
word_seg2 = temp_seg
pos2 = temp_pos
text_data2_dict = temp_data_dict
length2 = temp_length
print 'deleting unnecessary variables ...'
del temp_data
del temp_data_bi
del temp_data_tri
del temp_seg
del temp_pos
del temp_length
del temp_data_dict
del seed
print 'begin batching and shuffle the order of batches...'


#Batching
train_batch = batch_make_dict(text_data[3000:],text_data_bi[3000:],text_data_tri[3000:],word_seg[3000:],pos[3000:],text_data_dict[3000:],length[3000:],batch_size)


seed = np.random.permutation(len(train_batch))
temp_batch = []
for item in seed:
    temp_batch.append(train_batch[item])
train_batch = temp_batch
valid_batch = batch_make_dict(text_data[:3000],text_data_bi[:3000],text_data_tri[:3000],word_seg[:3000],pos[:3000],text_data_dict[:3000],length[:3000],batch_size)


train_batch2 = batch_make_dict(text_data2[10000:],text_data2_bi[10000:],text_data2_tri[10000:],word_seg2[10000:],pos2[10000:],text_data2_dict[10000:],length2[10000:],batch_size)
seed = np.random.permutation(len(train_batch2))
temp_batch = []
for item in seed:
    temp_batch.append(train_batch2[item])
train_batch2 = temp_batch
del temp_batch
del seed
valid_batch2 = batch_make_dict(text_data2[:10000],text_data2_bi[:10000],text_data2_tri[:10000],word_seg2[:10000],pos2[:10000],text_data2_dict[:10000],length2[:10000],batch_size)
vocab_s = len(text_token)+1

print 'Character dictionary size:'
print (vocab_s)
seed = np.random.permutation(len(text_data3))
temp_data = []
temp_data_bi = []
temp_data_tri = []
temp_seg = []
temp_pos = []
temp_length = []
temp_data_dict = []
for item in seed:
    print item
    temp_data.append(text_data3[item])
    temp_data_bi.append(text_data3_bi[item])
    temp_data_tri.append(text_data3_tri[item])
    temp_seg.append(word_seg3[item])
    temp_pos.append(pos3[item])
    temp_length.append(length3[item])
    temp_data_dict.append(text_data3_dict[item])
text_data3 = temp_data
text_data3_bi = temp_data_bi
text_data3_tri = temp_data_tri
word_seg3 = temp_seg
pos3 = temp_pos
text_data3_dict = temp_data_dict
length3 = temp_length
train_batch3 = batch_make_dict(text_data3[1000:],text_data3_bi[1000:],text_data3_tri[1000:],word_seg3[1000:],pos3[1000:],text_data3_dict[1000:],length3[1000:],batch_size)
valid_batch3 = batch_make_dict(text_data3[:1000],text_data3_bi[:1000],text_data3_tri[:1000],word_seg3[:1000],pos3[:1000],text_data3_dict[:1000],length3[:1000],batch_size)
vocab_s = len(text_token)+1

#print pos[3000]

####################Core!!      
#Define Model 
print 'Tensorflow model building ..'
with tf.Graph().as_default() as graph:
    # Add the data to the TensorFlow graph.
    x_t = tf.placeholder(tf.int32, [None, None])
    x_t1 = tf.placeholder(tf.int32,[None,None])
    x_t2 = tf.placeholder(tf.int32,[None,None])
    y_cws_t = tf.placeholder(tf.int32, [None, None])
    x_cws_t = tf.placeholder(tf.float32,[None,None,num_class_cws])
    y_pos_t = tf.placeholder(tf.int32,[None,None])
    x_dict_t = tf.placeholder(tf.float32,[None,None,n_dict_feature])
    sequence_lengths_t = tf.placeholder(tf.int32, [None])
    dict_x_t = tf.placeholder(tf.int32,[None,None])
    keep_prob = tf.placeholder(tf.float32,[1])

    #Share
    #embeddinglayer unigram + bigram + trigram
    embedding_share = tf.get_variable("embedding_share", [vocab_s, embedding_size], trainable=True)
    embedding_share_bi = tf.get_variable('embedding_share_bi',[vocab_s,embedding_size],trainable=True)
    embedding_share_tri = tf.get_variable('embedding_share_tri',[vocab_s,embedding_size],trainable=True)
     
    embedding_input = tf.nn.embedding_lookup(embedding_share, x_t)
    embedding_input_t1_0 = tf.nn.embedding_lookup(embedding_share_bi, x_t)
    embedding_input_t1_1 = tf.nn.embedding_lookup(embedding_share_bi, x_t1)
    embedding_input_t2_0 = tf.nn.embedding_lookup(embedding_share_tri,x_t)
    embedding_input_t2_1 = tf.nn.embedding_lookup(embedding_share_tri,x_t1)
    embedding_input_t2_2 = tf.nn.embedding_lookup(embedding_share_tri,x_t2)    
    embeddings_input = tf.concat([embedding_input,embedding_input_t1_0,embedding_input_t1_1,embedding_input_t2_2,embedding_input_t2_0,embedding_input_t2_1],-1)
#    embedding_share = tf.Variable(tf.random_normal([vocab_s, embedding_size], stddev=0.35),name="embedding_share")
    with tf.variable_scope('share'):#Shared/LSTM
        
        with tf.variable_scope('LSTM'):
            cell_fw_share = tf.contrib.rnn.GRUCell(n_hidden)
            cell_bw_share = tf.contrib.rnn.GRUCell(n_hidden)
            _initial_state_fw_share = cell_fw_share.zero_state(num_examples, tf.float32)
            _initial_state_bw_share = cell_bw_share.zero_state(num_examples, tf.float32)
            cell_fw_share = tf.contrib.rnn.DropoutWrapper(cell_fw_share,output_keep_prob=keep_prob[0])
            cell_bw_share = tf.contrib.rnn.DropoutWrapper(cell_bw_share,output_keep_prob=keep_prob[0])        
            outputs_share, state_share = tf.nn.bidirectional_dynamic_rnn(cell_fw_share, cell_bw_share, embedding_input, time_major=False, parallel_iterations=100,sequence_length=sequence_lengths_t, initial_state_fw=_initial_state_fw_share,initial_state_bw=_initial_state_bw_share,scope='share')
            outputs_share = tf.concat(outputs_share, 2)
        


    #CWS
    with tf.variable_scope('cws_CTB'):# CTB/LSTM
        

        with tf.variable_scope('LSTM'):
            cell_fw_cws = tf.contrib.rnn.GRUCell(n_hidden)
            cell_bw_cws = tf.contrib.rnn.GRUCell(n_hidden)
            _initial_state_fw_cws = cell_fw_cws.zero_state(num_examples, tf.float32)
            _initial_state_bw_cws = cell_bw_cws.zero_state(num_examples, tf.float32)
            cell_fw_cws = tf.contrib.rnn.DropoutWrapper(cell_fw_cws, output_keep_prob=keep_prob[0])
            cell_bw_cws = tf.contrib.rnn.DropoutWrapper(cell_bw_cws, output_keep_prob=keep_prob[0])
            outputs_cws, state_cws = tf.nn.bidirectional_dynamic_rnn(cell_fw_cws, cell_bw_cws, embedding_input, time_major=False, parallel_iterations=100,sequence_length=sequence_lengths_t, initial_state_fw=_initial_state_fw_cws,initial_state_bw=_initial_state_bw_cws,scope='cws')
            outputs_cws = tf.concat(outputs_cws, 2)
        
        # CRF
        weights_cws = tf.get_variable("weights_cws", [2*num_features+n_dict_feature, num_class_cws]) 
        transition_params_cws = tf.get_variable("trans_mat_cws", [num_class_cws, num_class_cws]) 
        outputs_cws_share = tf.concat([outputs_cws,outputs_share,x_dict_t], -1)                         
        matricized_x_t_cws = tf.reshape(outputs_cws_share, [-1, 2*num_features+n_dict_feature])
        matricized_unary_scores_cws = tf.matmul(matricized_x_t_cws, weights_cws)
        unary_scores_cws = tf.reshape(matricized_unary_scores_cws,[num_examples, num_words, num_class_cws])     
        log_likelihood_cws, transition_params_cws = tf.contrib.crf.crf_log_likelihood(unary_scores_cws, y_cws_t, sequence_lengths_t,transition_params=transition_params_cws)
        
    
    #PoS
    with tf.variable_scope('pos_CTB'):
        
    #    embedding_input_pos_drop = tf.nn.dropout(embedding_input_pos,keep_prob[0])
        with tf.variable_scope('LSTM'):           
            cell_fw_pos = tf.contrib.rnn.GRUCell(n_hidden)
            cell_bw_pos = tf.contrib.rnn.GRUCell(n_hidden)
            _initial_state_fw_pos = cell_fw_pos.zero_state(num_examples, tf.float32)
            _initial_state_bw_pos = cell_bw_pos.zero_state(num_examples, tf.float32)
            cell_fw_pos = tf.contrib.rnn.DropoutWrapper(cell_fw_pos, output_keep_prob=keep_prob[0])
            cell_bw_pos = tf.contrib.rnn.DropoutWrapper(cell_bw_pos, output_keep_prob=keep_prob[0])    
            outputs_pos, state_pos = tf.nn.bidirectional_dynamic_rnn(cell_fw_pos, cell_bw_pos, embedding_input, time_major=False, parallel_iterations=100,sequence_length=sequence_lengths_t, initial_state_fw=_initial_state_fw_pos,initial_state_bw=_initial_state_bw_pos,scope='pos')
            outputs_pos = tf.concat(outputs_pos, 2)
        
        #CRF 
        weights_pos = tf.get_variable("weights_pos", [2*num_features, num_class_pos]) 
        transition_params_pos = tf.get_variable("trans_mat_pos", [num_class_pos, num_class_pos]) 
        outputs_pos_share = tf.concat([outputs_pos,outputs_share], -1)                  
        matricized_x_t_pos = tf.reshape(outputs_pos_share, [-1, 2*num_features])  
        matricized_unary_scores_pos = tf.matmul(matricized_x_t_pos, weights_pos)    
        unary_scores_pos = tf.reshape(matricized_unary_scores_pos,[num_examples, num_words, num_class_pos])     
        log_likelihood_pos, transition_params_pos = tf.contrib.crf.crf_log_likelihood(unary_scores_pos, y_pos_t, sequence_lengths_t,transition_params=transition_params_pos)
        
    loss_cws = tf.reduce_mean(-log_likelihood_cws)
    loss_pos = tf.reduce_mean(-log_likelihood_pos)

    train_op = tf.train.AdamOptimizer(0.001).minimize(loss_pos)
    train_op2 = tf.train.AdamOptimizer(0.001).minimize(loss_cws)
    
    #2014 standard
    with tf.variable_scope('CWS_PKU2014'):

        with tf.variable_scope('LSTM'):
            cell_fw_cws_2 = tf.contrib.rnn.GRUCell(n_hidden)
            cell_bw_cws_2 = tf.contrib.rnn.GRUCell(n_hidden)
            _initial_state_fw_cws_2 = cell_fw_cws.zero_state(num_examples, tf.float32)
            _initial_state_bw_cws_2 = cell_bw_cws.zero_state(num_examples, tf.float32)
            cell_fw_cws_2 = tf.contrib.rnn.DropoutWrapper(cell_fw_cws_2, output_keep_prob=keep_prob[0])
            cell_bw_cws_2 = tf.contrib.rnn.DropoutWrapper(cell_bw_cws_2, output_keep_prob=keep_prob[0])
            outputs_cws_2, state_cws_2 = tf.nn.bidirectional_dynamic_rnn(cell_fw_cws_2, cell_bw_cws_2, embedding_input, time_major=False, parallel_iterations=100,sequence_length=sequence_lengths_t, initial_state_fw=_initial_state_fw_cws_2,initial_state_bw=_initial_state_bw_cws_2,scope='cws2')
            outputs_cws_2 = tf.concat(outputs_cws_2, 2)
        
        weights_cws_2 = tf.get_variable("weights_cws_2", [2*num_features+n_dict_feature, num_class_cws]) 
        transition_params_cws_2 = tf.get_variable("trans_mat_cws_2", [num_class_cws, num_class_cws])
#        x_dict_t_3d2 = tf.reshape(x_dict_t,[num_examples,num_words,1])
        
        outputs_cws_share_2 = tf.concat([outputs_cws_2,outputs_share,x_dict_t], -1)                         
        matricized_x_t_cws_2 = tf.reshape(outputs_cws_share_2, [-1, 2*num_features+n_dict_feature])
        matricized_unary_scores_cws_2 = tf.matmul(matricized_x_t_cws_2, weights_cws_2)
        unary_scores_cws_2 = tf.reshape(matricized_unary_scores_cws_2,[num_examples, num_words, num_class_cws])     
        log_likelihood_cws_2, transition_params_cws_2 = tf.contrib.crf.crf_log_likelihood(unary_scores_cws_2, y_cws_t, sequence_lengths_t,transition_params=transition_params_cws_2)
        
    
    #PoS
    with tf.variable_scope('pos_2014PKU'):

        with tf.variable_scope('LSTM'):
            cell_fw_pos_2 = tf.contrib.rnn.GRUCell(n_hidden)
            cell_bw_pos_2 = tf.contrib.rnn.GRUCell(n_hidden)
            _initial_state_fw_pos_2 = cell_fw_pos.zero_state(num_examples, tf.float32)
            _initial_state_bw_pos_2 = cell_bw_pos.zero_state(num_examples, tf.float32)
            cell_fw_pos_2 = tf.contrib.rnn.DropoutWrapper(cell_fw_pos_2, output_keep_prob=keep_prob[0])
            cell_bw_pos_2 = tf.contrib.rnn.DropoutWrapper(cell_bw_pos_2, output_keep_prob=keep_prob[0])    
            outputs_pos_2, state_pos_2 = tf.nn.bidirectional_dynamic_rnn(cell_fw_pos_2, cell_bw_pos_2, embedding_input, time_major=False, parallel_iterations=100,sequence_length=sequence_lengths_t, initial_state_fw=_initial_state_fw_pos_2,initial_state_bw=_initial_state_bw_pos_2,scope='pos2')
            outputs_pos_2 = tf.concat(outputs_pos_2, 2)
        
        weights_pos_2 = tf.get_variable("weights_pos_2", [2*num_features, num_class_pos_2]) 
        transition_params_pos_2 = tf.get_variable("trans_mat_pos_2", [num_class_pos_2, num_class_pos_2]) 
        outputs_pos_share_2 = tf.concat([outputs_pos_2,outputs_share], -1)                  
        matricized_x_t_pos_2 = tf.reshape(outputs_pos_share_2, [-1, 2*num_features])  
        matricized_unary_scores_pos_2 = tf.matmul(matricized_x_t_pos_2, weights_pos_2)    
        unary_scores_pos_2 = tf.reshape(matricized_unary_scores_pos_2,[num_examples, num_words, num_class_pos_2])     
        log_likelihood_pos_2, transition_params_pos_2 = tf.contrib.crf.crf_log_likelihood(unary_scores_pos_2, y_pos_t, sequence_lengths_t,transition_params=transition_params_pos_2)
        
    loss_cws_2 = tf.reduce_mean(-log_likelihood_cws_2)
    loss_pos_2 = tf.reduce_mean(-log_likelihood_pos_2)

    train_op3 = tf.train.AdamOptimizer(0.001).minimize(loss_pos_2)
    train_op4 = tf.train.AdamOptimizer(0.001).minimize(loss_cws_2)
    
    # LTP standard
    with tf.variable_scope('CWS_LTP'):
        with tf.variable_scope('LSTM'):
            cell_fw_cws_3 = tf.contrib.rnn.GRUCell(n_hidden)
            cell_bw_cws_3 = tf.contrib.rnn.GRUCell(n_hidden)
            _initial_state_fw_cws_3 = cell_fw_cws.zero_state(num_examples, tf.float32)
            _initial_state_bw_cws_3 = cell_bw_cws.zero_state(num_examples, tf.float32)
            cell_fw_cws_3 = tf.contrib.rnn.DropoutWrapper(cell_fw_cws_3, output_keep_prob=keep_prob[0])
            cell_bw_cws_3 = tf.contrib.rnn.DropoutWrapper(cell_bw_cws_3, output_keep_prob=keep_prob[0])
            outputs_cws_3, state_cws_3 = tf.nn.bidirectional_dynamic_rnn(cell_fw_cws_3, cell_bw_cws_3, embedding_input, time_major=False, parallel_iterations=100,sequence_length=sequence_lengths_t, initial_state_fw=_initial_state_fw_cws_3,initial_state_bw=_initial_state_bw_cws_3,scope='cws3')
            outputs_cws_3 = tf.concat(outputs_cws_3, 2)
    
        weights_cws_3 = tf.get_variable("weights_cws_3", [2*num_features+n_dict_feature, num_class_cws]) 
        transition_params_cws_3 = tf.get_variable("trans_mat_cws_3", [num_class_cws, num_class_cws])
        outputs_cws_share_3 = tf.concat([outputs_cws_3,outputs_share,x_dict_t], -1)                         
        matricized_x_t_cws_3 = tf.reshape(outputs_cws_share_3, [-1, 2*num_features+n_dict_feature])
        matricized_unary_scores_cws_3 = tf.matmul(matricized_x_t_cws_3, weights_cws_3)
        unary_scores_cws_3 = tf.reshape(matricized_unary_scores_cws_3,[num_examples, num_words, num_class_cws])     
        log_likelihood_cws_3, transition_params_cws_3 = tf.contrib.crf.crf_log_likelihood(unary_scores_cws_3, y_cws_t, sequence_lengths_t,transition_params=transition_params_cws_3)
        
    

    with tf.variable_scope('pos_LTP'):
        with tf.variable_scope('LSTM'):
            
            cell_fw_pos_3 = tf.contrib.rnn.GRUCell(n_hidden)
            cell_bw_pos_3 = tf.contrib.rnn.GRUCell(n_hidden)
            _initial_state_fw_pos_3 = cell_fw_pos.zero_state(num_examples, tf.float32)
            _initial_state_bw_pos_3 = cell_bw_pos.zero_state(num_examples, tf.float32)
            cell_fw_pos_3 = tf.contrib.rnn.DropoutWrapper(cell_fw_pos_3, output_keep_prob=keep_prob[0])
            cell_bw_pos_3 = tf.contrib.rnn.DropoutWrapper(cell_bw_pos_3, output_keep_prob=keep_prob[0])    
            outputs_pos_3, state_pos_3 = tf.nn.bidirectional_dynamic_rnn(cell_fw_pos_3, cell_bw_pos_3, embedding_input, time_major=False, parallel_iterations=100,sequence_length=sequence_lengths_t, initial_state_fw=_initial_state_fw_pos_3,initial_state_bw=_initial_state_bw_pos_3,scope='pos2')
            outputs_pos_3 = tf.concat(outputs_pos_3, 2)
        
        weights_pos_3 = tf.get_variable("weights_pos_3", [2*num_features, num_class_pos_3]) 
        transition_params_pos_3 = tf.get_variable("trans_mat_pos_3", [num_class_pos_3, num_class_pos_3]) 
        outputs_pos_share_3 = tf.concat([outputs_pos_3,outputs_share], -1)                  
        matricized_x_t_pos_3 = tf.reshape(outputs_pos_share_3, [-1, 2*num_features])  
        matricized_unary_scores_pos_3 = tf.matmul(matricized_x_t_pos_3, weights_pos_3)    
        unary_scores_pos_3 = tf.reshape(matricized_unary_scores_pos_3,[num_examples, num_words, num_class_pos_3])     
        log_likelihood_pos_3, transition_params_pos_3 = tf.contrib.crf.crf_log_likelihood(unary_scores_pos_3, y_pos_t, sequence_lengths_t,transition_params=transition_params_pos_3)
        
    loss_cws_3 = tf.reduce_mean(-log_likelihood_cws_3)
    loss_pos_3 = tf.reduce_mean(-log_likelihood_pos_3)

    train_op5 = tf.train.AdamOptimizer(0.001).minimize(loss_pos_3)
    train_op6 = tf.train.AdamOptimizer(0.001).minimize(loss_cws_3)
    
    init = tf.global_variables_initializer()#initize all variables





#train and validation
print 'training begins'
epoch = 100

with tf.Session(graph=graph) as sess:
    # Train for a fixed number of iterations.
    sess.run(init)
    writer = tf.summary.FileWriter('tensorboard_log', sess.graph)
    sess.run(tf.assign(embedding_share_bi,pretrain_embed))
    sess.run(tf.assign(embedding_share,pretrain_embed))
    sess.run(tf.assign(embedding_share_tri,pretrain_embed))
    saver = tf.train.Saver(max_to_keep=epoch)
#    saver.restore(sess,'model/with_share_model_Uni_bigram200_95.1730495855.ckpt')
#   Train
    for cur_epoch in range(epoch):
        for i in range(len(train_batch)):
            tf_unary_scores_cws, tf_transition_params_cws,_ = sess.run([ unary_scores_cws, transition_params_cws, train_op2],feed_dict={x_t:train_batch[i][0],x_t1:train_batch[i][1],x_t2:train_batch[i][2],y_cws_t:train_batch[i][3],x_dict_t:train_batch[i][5],sequence_lengths_t:train_batch[i][6],keep_prob:[0.8]})            
#            tf_unary_scores_pos, tf_transition_params_pos,_ = sess.run([ unary_scores_pos, transition_params_pos, train_op],feed_dict={x_t:train_batch[i][0],x_t1:train_batch[i][1],x_cws_t:tf_unary_scores_cws,y_pos_t:train_batch[i][3],sequence_lengths_t:train_batch[i][5],keep_prob:[0.8]})            
            assert not (True in np.isnan(tf_unary_scores_cws))
#            assert not (True in np.isnan(tf_unary_scores_pos))
            if i % 100 == 0:
                correct_labels_cws = 0
#                correct_labels_pos = 0
                total_labels = 0
                for tf_unary_scores_cws_, y_cws, y_pos, sequence_length_ in zip(tf_unary_scores_cws, train_batch[i][3],train_batch[i][4],train_batch[i][6]):
              # Remove padding from the scores and tag sequence.
                    tf_unary_scores_cws_ = tf_unary_scores_cws_[:sequence_length_]
                    y_cws = y_cws[:sequence_length_]
#                    tf_unary_scores_pos_ = tf_unary_scores_pos_[:sequence_length_]
#                    y_pos = y_pos[:sequence_length_]
                    
              # Compute the highest scoring sequence.

                    viterbi_sequence_cws, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_cws_, tf_transition_params_cws)
#                    viterbi_sequence_pos, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_pos_, tf_transition_params_pos)
              # Evaluate word-level accuracy.
#                    print viterbi_sequence
                    correct_labels_cws += np.sum(np.equal(viterbi_sequence_cws, y_cws))
#                    correct_labels_pos += np.sum(np.equal(viterbi_sequence_pos, y_pos))
                    total_labels += sequence_length_
                accuracy_cws = 100.0 * correct_labels_cws / float(total_labels)
#                accuracy_pos = 100.0 * correct_labels_pos / float(total_labels)
                print("CWS_Accuracy: %.2f%%" % accuracy_cws)
#                print("POS_Accuracy: %.2f%%" % accuracy_pos)
        print ('2014 begins:')
        for i in range(len(train_batch2)):

            tf_unary_scores_cws, tf_transition_params_cws,_ = sess.run([ unary_scores_cws_2, transition_params_cws_2, train_op4],feed_dict={x_t:train_batch2[i][0],x_t1:train_batch2[i][1],x_t2:train_batch2[i][2],y_cws_t:train_batch2[i][3],x_dict_t:train_batch2[i][5],sequence_lengths_t:train_batch2[i][6],keep_prob:[0.8]})            
#            tf_unary_scores_pos, tf_transition_params_pos,_ = sess.run([ unary_scores_pos_2, transition_params_pos_2, train_op3],feed_dict={x_t:train_batch2[i][0],x_cws_t:tf_unary_scores_cws,y_pos_t:train_batch2[i][2],sequence_lengths_t:train_batch2[i][4],keep_prob:[0.8]})            
            assert not (True in np.isnan(tf_unary_scores_cws))
#            assert not (True in np.isnan(tf_unary_scores_pos))
            if i % 100 == 0:
                correct_labels_cws = 0
                correct_labels_pos = 0
                total_labels = 0
                for tf_unary_scores_cws_, y_cws, y_pos, sequence_length_ in zip(tf_unary_scores_cws, train_batch2[i][3],train_batch2[i][4],train_batch2[i][6]):
              # Remove padding from the scores and tag sequence.
                    tf_unary_scores_cws_ = tf_unary_scores_cws_[:sequence_length_]
                    y_cws = y_cws[:sequence_length_]
#                    tf_unary_scores_pos_ = tf_unary_scores_pos_[:sequence_length_]
#                    y_pos = y_pos[:sequence_length_]
                    
              # Compute the highest scoring sequence.

                    viterbi_sequence_cws, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_cws_, tf_transition_params_cws)
#                    viterbi_sequence_pos, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_pos_, tf_transition_params_pos)
              # Evaluate word-level accuracy.
                    correct_labels_cws += np.sum(np.equal(viterbi_sequence_cws, y_cws))
#                    correct_labels_pos += np.sum(np.equal(viterbi_sequence_pos, y_pos))
                    total_labels += sequence_length_
                accuracy_cws = 100.0 * correct_labels_cws / float(total_labels)
#                accuracy_pos = 100.0 * correct_labels_pos / float(total_labels)
                print("CWS_Accuracy: %.2f%%" % accuracy_cws)
#                print("POS_Accuracy: %.2f%%" % accuracy_pos)
        print 'LTP begins:'        
        for i in range(len(train_batch3)):

            tf_unary_scores_cws, tf_transition_params_cws,_ = sess.run([ unary_scores_cws_3, transition_params_cws_3, train_op6],feed_dict={x_t:train_batch3[i][0],x_t1:train_batch3[i][1],x_t2:train_batch3[i][2],y_cws_t:train_batch3[i][3],x_dict_t:train_batch3[i][5],sequence_lengths_t:train_batch3[i][6],keep_prob:[0.8]})            
#            tf_unary_scores_pos, tf_transition_params_pos,_ = sess.run([ unary_scores_pos_2, transition_params_pos_2, train_op3],feed_dict={x_t:train_batch2[i][0],x_cws_t:tf_unary_scores_cws,y_pos_t:train_batch2[i][2],sequence_lengths_t:train_batch2[i][4],keep_prob:[0.8]})            
            assert not (True in np.isnan(tf_unary_scores_cws))
#            assert not (True in np.isnan(tf_unary_scores_pos))
            if i % 100 == 0:
                correct_labels_cws = 0
                correct_labels_pos = 0
                total_labels = 0
                for tf_unary_scores_cws_, y_cws, y_pos, sequence_length_ in zip(tf_unary_scores_cws, train_batch3[i][3],train_batch3[i][4],train_batch3[i][6]):
              # Remove padding from the scores and tag sequence.
                    tf_unary_scores_cws_ = tf_unary_scores_cws_[:sequence_length_]
                    y_cws = y_cws[:sequence_length_]
#                    tf_unary_scores_pos_ = tf_unary_scores_pos_[:sequence_length_]
#                    y_pos = y_pos[:sequence_length_]
                    
              # Compute the highest scoring sequence.

                    viterbi_sequence_cws, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_cws_, tf_transition_params_cws)
#                    viterbi_sequence_pos, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_pos_, tf_transition_params_pos)
              # Evaluate word-level accuracy.
                    correct_labels_cws += np.sum(np.equal(viterbi_sequence_cws, y_cws))
#                    correct_labels_pos += np.sum(np.equal(viterbi_sequence_pos, y_pos))
                    total_labels += sequence_length_
                accuracy_cws = 100.0 * correct_labels_cws / float(total_labels)
#                accuracy_pos = 100.0 * correct_labels_pos / float(total_labels)
                print("CWS_Accuracy: %.2f%%" % accuracy_cws)
#                print("POS_Accuracy: %.2f%%" % accuracy_pos) 
        correct_labels_cws = 0
        correct_labels_pos = 0
        total_labels = 0

        for i in range(len(valid_batch)):
            tf_unary_scores_cws, tf_transition_params_cws = sess.run([ unary_scores_cws, transition_params_cws],feed_dict={x_t:valid_batch[i][0],x_t1:valid_batch[i][1],x_t2:valid_batch[i][2],y_cws_t:valid_batch[i][3],x_dict_t:valid_batch[i][5],sequence_lengths_t:valid_batch[i][6],keep_prob:[1.0]})
#            tf_unary_scores_pos, tf_transition_params_pos = sess.run([ unary_scores_pos, transition_params_pos],feed_dict={x_t:valid_batch[i][0],x_cws_t:tf_unary_scores_cws,y_pos_t:valid_batch[i][2],sequence_lengths_t:valid_batch[i][4],keep_prob:[1.0]})     
            
            for tf_unary_scores_cws_, y_cws, y_pos, sequence_length_ in zip(tf_unary_scores_cws, valid_batch[i][3],valid_batch[i][4],valid_batch[i][6]):#most possible path searching
              # Remove padding from the scores and tag sequence.
                tf_unary_scores_cws_ = tf_unary_scores_cws_[:sequence_length_]
                y_cws = y_cws[:sequence_length_]
#                tf_unary_scores_pos_ = tf_unary_scores_pos_[:sequence_length_]
#                y_pos = y_pos[:sequence_length_]
          # Compute the highest scoring sequence.
                viterbi_sequence_cws, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_cws_, tf_transition_params_cws)
#                viterbi_sequence_pos, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_pos_, tf_transition_params_pos)
          # Evaluate word-level accuracy.
#                    print viterbi_sequence
                correct_labels_cws += np.sum(np.equal(viterbi_sequence_cws, y_cws))
#                correct_labels_pos += np.sum(np.equal(viterbi_sequence_pos, y_pos))
                total_labels += sequence_length_
        accuracy_cws = 100.0 * correct_labels_cws / float(total_labels)
#        accuracy_pos = 100.0 * correct_labels_pos / float(total_labels)
        print('epoch:' +str(cur_epoch))
        print("valid_CWS_Accuracy: %.2f%%" % accuracy_cws)
#        print("valid_POS_Accuracy: %.2f%%" % accuracy_pos)
#'''
#        correct_labels_cws = 0
#        correct_labels_pos = 0
#        total_labels = 0
#        for i in range(len(valid_batch2)):
#            tf_unary_scores_cws, tf_transition_params_cws = sess.run([ unary_scores_cws_2, transition_params_cws_2],feed_dict={x_t:valid_batch2[i][0],x_t1:valid_batch2[i][1],y_cws_t:valid_batch2[i][2],x_dict_t:valid_batch2[i][4],sequence_lengths_t:valid_batch2[i][5],keep_prob:[1.0]})
##            tf_unary_scores_pos, tf_transition_params_pos = sess.run([ unary_scores_pos_2, transition_params_pos_2],feed_dict={x_t:valid_batch2[i][0],x_cws_t:tf_unary_scores_cws,y_pos_t:valid_batch2[i][2],sequence_lengths_t:valid_batch2[i][4],keep_prob:[1.0]})     
#            
#            for tf_unary_scores_cws_, y_cws, y_pos, sequence_length_ in zip(tf_unary_scores_cws, valid_batch2[i][2],valid_batch2[i][3],valid_batch2[i][5]):
#              # Remove padding from the scores and tag sequence.
#                tf_unary_scores_cws_ = tf_unary_scores_cws_[:sequence_length_]
#                y_cws = y_cws[:sequence_length_]
##                tf_unary_scores_pos_ = tf_unary_scores_pos_[:sequence_length_]
##                y_pos = y_pos[:sequence_length_]
#          # Compute the highest scoring sequence.
#                viterbi_sequence_cws, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_cws_, tf_transition_params_cws)
##                viterbi_sequence_pos, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_pos_, tf_transition_params_pos)
#          # Evaluate word-level accuracy.
##                    print viterbi_sequence
#                correct_labels_cws += np.sum(np.equal(viterbi_sequence_cws, y_cws))
##                correct_labels_pos += np.sum(np.equal(viterbi_sequence_pos, y_pos))
#                total_labels += sequence_length_
#        accuracy_cws = 100.0 * correct_labels_cws / float(total_labels)
##        accuracy_pos = 100.0 * correct_labels_pos / float(total_labels)
#        print('epoch:' +str(cur_epoch))
#        print("valid_CWS_Accuracy: %.2f%%" % accuracy_cws)
##        print("valid_POS_Accuracy: %.2f%%" % accuracy_pos)
##        train_batch = batch_make_dict(text_data[3000:],text_data_bi[3000:],word_seg[3000:],pos[3000:],text_data_dict[3000:],length[3000:],batch_size)
##        train_batch2 = batch_make_dict(text_data2[10000:],text_data2_bi[10000:],word_seg2[10000:],pos2[10000:],text_data2_dict[10000:],length2[10000:],batch_size)       
#'''
#        for i in range(len(valid_batch3)):
#            tf_unary_scores_cws, tf_transition_params_cws = sess.run([ unary_scores_cws_3, transition_params_cws_3],feed_dict={x_t:valid_batch3[i][0],x_t1:valid_batch3[i][1],y_cws_t:valid_batch3[i][2],x_dict_t:valid_batch3[i][4],sequence_lengths_t:valid_batch3[i][5],keep_prob:[1.0]})
##            tf_unary_scores_pos, tf_transition_params_pos = sess.run([ unary_scores_pos_2, transition_params_pos_2],feed_dict={x_t:valid_batch2[i][0],x_cws_t:tf_unary_scores_cws,y_pos_t:valid_batch2[i][2],sequence_lengths_t:valid_batch2[i][4],keep_prob:[1.0]})     
#            
#            for tf_unary_scores_cws_, y_cws, y_pos, sequence_length_ in zip(tf_unary_scores_cws, valid_batch3[i][2],valid_batch3[i][3],valid_batch3[i][5]):
#              # Remove padding from the scores and tag sequence.
#                tf_unary_scores_cws_ = tf_unary_scores_cws_[:sequence_length_]
#                y_cws = y_cws[:sequence_length_]
##                tf_unary_scores_pos_ = tf_unary_scores_pos_[:sequence_length_]
##                y_pos = y_pos[:sequence_length_]
#          # Compute the highest scoring sequence.
#                viterbi_sequence_cws, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_cws_, tf_transition_params_cws)
##                viterbi_sequence_pos, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_pos_, tf_transition_params_pos)
#          # Evaluate word-level accuracy.
##                    print viterbi_sequence
#                correct_labels_cws += np.sum(np.equal(viterbi_sequence_cws, y_cws))
##                correct_labels_pos += np.sum(np.equal(viterbi_sequence_pos, y_pos))
#                total_labels += sequence_length_
#        accuracy_cws = 100.0 * correct_labels_cws / float(total_labels)
##        accuracy_pos = 100.0 * correct_labels_pos / float(total_labels)
#        print('epoch:' +str(cur_epoch))
#        print("valid_CWS_Accuracy: %.2f%%" % accuracy_cws)
#        print("valid_POS_Accuracy: %.2f%%" % accuracy_pos)
        correct_labels_cws = 0
        correct_labels_pos = 0
        total_labels = 0

        for i in range(len(valid_batch3)):
            tf_unary_scores_cws, tf_transition_params_cws = sess.run([ unary_scores_cws, transition_params_cws],feed_dict={x_t:valid_batch3[i][0],x_t1:valid_batch3[i][1],x_t2:valid_batch3[i][2],y_cws_t:valid_batch3[i][3],x_dict_t:valid_batch3[i][5],sequence_lengths_t:valid_batch3[i][6],keep_prob:[1.0]})
#            tf_unary_scores_pos, tf_transition_params_pos = sess.run([ unary_scores_pos, transition_params_pos],feed_dict={x_t:valid_batch[i][0],x_cws_t:tf_unary_scores_cws,y_pos_t:valid_batch[i][2],sequence_lengths_t:valid_batch[i][4],keep_prob:[1.0]})     
            
            for tf_unary_scores_cws_, y_cws, y_pos, sequence_length_ in zip(tf_unary_scores_cws, valid_batch3[i][3],valid_batch3[i][4],valid_batch3[i][6]):#most possible path searching
              # Remove padding from the scores and tag sequence.
                tf_unary_scores_cws_ = tf_unary_scores_cws_[:sequence_length_]
                y_cws = y_cws[:sequence_length_]
#                tf_unary_scores_pos_ = tf_unary_scores_pos_[:sequence_length_]
#                y_pos = y_pos[:sequence_length_]
          # Compute the highest scoring sequence.
                viterbi_sequence_cws, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_cws_, tf_transition_params_cws)
#                viterbi_sequence_pos, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_pos_, tf_transition_params_pos)
          # Evaluate word-level accuracy.
#                    print viterbi_sequence
                correct_labels_cws += np.sum(np.equal(viterbi_sequence_cws, y_cws))
#                correct_labels_pos += np.sum(np.equal(viterbi_sequence_pos, y_pos))
                total_labels += sequence_length_
        accuracy_cws = 100.0 * correct_labels_cws / float(total_labels)
#        accuracy_pos = 100.0 * correct_labels_pos / float(total_labels)
        print('epoch:' +str(cur_epoch))
        print("valid_CWS_Accuracy: %.2f%%" % accuracy_cws)
        saver.save(sess,  'model/' + 'initilzed_with_share_model_Uni_bigram'+str(n_hidden)+'_' + str(accuracy_cws) + '.ckpt')
####################Core!! 

