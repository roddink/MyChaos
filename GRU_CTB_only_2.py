"""
Created on Tue Jun 13 17:33:29 2017

@author: sz_rafahao
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
#from utils import *
from pos_tag_preprocess import *
from lexicon_features_cws import *
from gensim.models.word2vec import Word2Vec
def file_2_dict(path):
    token_file = np.load(path)
    keys = []
    values = []
    for item,item2 in zip(token_file[0],token_file[1]):
        keys.append(item.decode('utf8').encode('utf8'))
        values.append(int(item2))
    return dict(zip(keys,values))
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
# Data settings.
num_examples = 100

num_features = 200

n_hidden= 100
embedding_size =300
gensim_model = Word2Vec.load('data/mongoDB_w2v_epoch5.txt')# init_embedding 
#gensim_path = 'data/mongoDB_w2v_epoch5.txt'
train_path = 'data/word_seg_data_963656sentences.txt'
#cmodel = CharEmbedding(gensim_path)
#global index2word
#index2word = cmodel.index_2_word()
#train_data, valid_data, vocab_s, max_len = data_loader(train_path, cmodel)
#vocab_s = cmodel.vocab_size()
max_len = 500
#import time
#text_data,text_token,word_seg,word_seg_token,pos,pos_token,length,num_words,joint,joint_token = Preprocess('data/CTB.txt',tokenized=False)
#text_data2,text_token2,word_seg2,word_seg_token2,pos2,pos_token2,length2,num_words2,joint2,joint_token2 = Preprocess('data/pku_2014.txt',tokenized=False)
#text_data_new = text_data+text_data2
#text_data_text = []
#text_data2_text = []
text_data,text_token,word_seg,word_seg_token,pos,pos_token,length,num_words,joint,joint_token = Preprocess('data/CTB.txt',tokenized=False)
text_token = np.load('CWS_word_embedding_CTB+2014.npy')
text_token = dict(zip(text_token[0,:],text_token[1,:]))
pretrain_embed,text_token,_= init_embedding(gensim_model,embedding_size)
#text_token = file_2_dict('CWS_word_embedding.npy')
#word_seg_token = file_2_dict('CWS_POS_cws_embedding.npy')
word_seg_token2 = file_2_dict('CWS_POS_cws2_embedding.npy')
#pos_token = file_2_dict('CWS_POS_pos_embedding.npy')
pos_token2 = file_2_dict('CWS_POS_pos2_embedding.npy')

#for ind in range(len(text_data)):
#    text_data_text.append(''.join(text_data[ind]))
#for ind in range(len(text_data2)):
#    text_data2_text.append(''.join(text_data2[ind]))
lm = lexicon_feat('data/dictionary2.txt')
text_data_dict = []
#for item in text_data_text:
##    start = time.clock()
#    text_data_dict.append(match(item,lm))
##    print (time.clock()-start)*1000
#text_data2_dict = []
#for item in text_data2_text:
#    text_data2_dict.append(match(item,lm))

#temp_text = []
#for item in text_data_new:
#    temp_text.extend(item.tolist())
#temp_text = list(set(temp_text))
#np.save('dict_text.npy',temp_text)
#text_token3 = {k: v+1 for v, k in enumerate(temp_text)}
#text_data = tokenizer(text_data,text_token,max_len)
#word_seg = tokenizer(word_seg,word_seg_token,max_len)
#pos = tokenizer(pos,pos_token,max_len)
#text_data2 = tokenizer(text_data2,text_token,max_len)
#word_seg2 = tokenizer(word_seg2,word_seg_token2,max_len)
#pos2 = tokenizer(pos2,pos_token2,max_len)
batch_size = 100
num_examples = batch_size
num_class_cws = len(word_seg_token)
#num_class_pos = len(pos_token)
num_class_pos = 43
num_class_pos_2 = len(pos_token2)
num_class_pos_3 = 20
print (word_seg_token)
print (word_seg_token2)
from langconv import *
def Sim_Chinese_2_Tra_Chinese(text):
    line = Converter('zh-hans').convert(text.decode('utf-8'))
    line = line.encode('utf-8')
    return line
def data_preprocess(text,vocab):
    text = Sim_Chinese_2_Tra_Chinese(text)
    text = text.decode('utf8')
    temp_text = np.zeros(500,dtype=int)
    for ind in range(min(len(text),max_len)):
        temp_text[ind]=vocab.get(text[ind].encode('utf8'),0)
    return temp_text.tolist(),min(len(text),500)
def run(token_text,dict_fea,length,sess):
#    text_input = np.zeros((100,1030))
#    dict_fea_input = np.zeros((100,1030))
#    length_input = np.zeros((100))
    text_t2 = bigram_maker(token_text)
    text_t3 = trigram_maker(token_text)
    token_text = np.array(token_text)
    text_t2 = np.array(text_t2)
    text_t3 = np.array(text_t3)
    dict_fea = np.array(dict_fea)
    length_input = np.array(length)
    if len(length) == 100:
        text_input = token_text
        text_input2 = text_t2
        text_input3 = text_t3
        dict_fea_input = dict_fea
        length_input = length
    else:
        text_input = np.zeros((100,500))
        text_input2 = np.zeros((100,500))
        text_input3 = np.zeros((100,500))
        dict_fea_input = np.zeros((100,500,3))
        length_input = np.zeros((100))
        text_input[:len(length)] = token_text
        text_input2[:len(length)] = text_t2
        text_input3[:len(length)] = text_t3
        dict_fea_input[:len(length)] = dict_fea
        length_input[:len(length)] = length
    tf_unary_scores, tf_transition_params = sess.run([ unary_scores_cws, transition_params_cws],feed_dict={x_t:text_input,x_t1:text_input2,x_t2:text_input3,x_dict_t:dict_fea_input,sequence_lengths_t:length_input,keep_prob:[1.0]})
    result = []
    score_list = []    
    for ind in range(len(length)):    
        tf_unary_scores_ = tf_unary_scores[ind,:length[ind]]
#    print tf_unary_scores
        viterbi_sequence, scores = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params)
        result.append(viterbi_sequence)
        score_list.append(scores/length[ind])
    return result,score_list
def visual(text,result):
    text = text.strip()
    text = text.decode('utf8')
    if len(text)> 500:
        text=text[:500]
    seg_text = ''

    for char,token in zip(text,result):
        if token==2 or token ==1:
            seg_text = seg_text+' '+char
        else:
            seg_text = seg_text + char
    
    return seg_text[1:].encode('utf8')

def batch_make(text_data,word_seg,pos,length,batch_size):
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
    
def batch_make_joint(text_data,joint,length,batch_size):
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

def batch_maker(data,batch_size):
    temp_ind = 0
    length_data = len(data[0])
    num_batch = length_data // batch_size + 1
    seed = np.random.permutation(len(data[0]))
    data_new = data
    for ind in range(len(data_new[0])):
        data[0][ind] = data_new[0][seed[ind]]
        data[1][ind] = data_new[1][seed[ind]]
    del data_new
    batches= []
    def get_label(temp_data,max_len):
        label = []
        sentence = []
        label_dict = dict({"S":0,"B":1,"M":2,"E":3})
        
        for ind in range(len(temp_data[0])):
            temp = np.zeros((max_len),dtype=int)
            temp_sen = np.zeros((max_len),dtype=int)
            for ind2 in range(len(temp_data[0][ind])):
                temp_sen[ind2] = temp_data[0][ind][ind2]
                temp[ind2] = label_dict[temp_data[1][ind][ind2]]
            label.append(temp)
            sentence.append(temp_sen)
        return sentence,label
    def get_length(temp_data):
        length = np.zeros((len(temp_data)),dtype=int)
        for ind in range(len(temp_data)):
            length[ind] = len(temp_data[ind])
        return length.tolist()
    length = get_length(data[0])
    temp = np.where(np.array(length)>0)
    np.save('length.npy',length)
    data_new = data
    for ind in range(len(temp[0])):
        data[0][ind] = data_new[0][temp[0][ind]]
        data[1][ind] = data_new[1][temp[0][ind]]
    data[0] = data[0][:len(temp[0])]
    data[1] = data[1][:len(temp[0])]
    length = get_length(data[0])
    length_data = len(data[0])
    num_batch = length_data // batch_size + 1
    del data_new
    sentences,label = get_label(data,max_len)
    for ind in range(num_batch-1):
        batches.append([sentences[temp_ind:temp_ind+batch_size],label[temp_ind:temp_ind+batch_size],[length[temp_ind:temp_ind+batch_size]]])
        temp_ind += batch_size
    return batches
def batch_make_dict(text_data,text_data_bi,text_data_tri,word_seg,pos,dictionary,length,batch_size):
    batch = []
    for ind in range(0,len(text_data),batch_size):
        batch.append([text_data[ind:ind+batch_size],text_data_bi[ind:ind+batch_size],text_data_tri[ind:ind+batch_size],word_seg[ind:ind+batch_size],pos[ind:ind+batch_size],dictionary[ind:ind+batch_size],length[ind:ind+batch_size]])
    return batch[:-1]

#train_batch = batch_make_dict(text_data[3000:],word_seg[3000:],pos[3000:],text_data_dict[3000:],length[3000:],batch_size)
#valid_batch = batch_make_dict(text_data[:3000],word_seg[:3000],pos[:3000],text_data_dict[:3000],length[:3000],batch_size)
#train_batch2 = batch_make_dict(text_data2[10000:],word_seg2[10000:],pos2[10000:],text_data2_dict[10000:],length2[10000:],batch_size)
#valid_batch2 = batch_make_dict(text_data2[:10000],word_seg2[:10000],pos2[:10000],text_data2_dict[:10000],length2[:10000],batch_size)
vocab_s = 17413
print (vocab_s)
#vocab_s2 = len(text_token2)+1
#print pos[3000]
epoch = 100
num_words=500
n_dict_feature = 3
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





import datetime
name = 'news_orig'
file = open(name+'.txt')
sample_text = file.readlines()

#seed = np.random.permutation(num_samples)
with tf.Session(graph=graph) as sess:
    # Train for a fixed number of iterations.
    sess.run(init)

#    sess.run(tf.assign(embedding,cmodel.embedding_matrix()))
    saver = tf.train.Saver()
    saver.restore(sess,'model/initilzed_with_share_model_Uni_bigram100_55.049162559.ckpt')
    result = []
    score_list = []
    text_list = []
    length_list = []
    dict_list = []
    file = open(name+'_seg.txt','w')
    a = datetime.datetime.now()
    for item in sample_text: 
        item = item.strip()        
        text,length_text = data_preprocess(item,text_token)
        
        text_list.append(text)
        length_list.append(length_text)
        dict_list.append(lm.produce_features(item,max_len=max_len))
    result_list = []    
    for ind in range(0,len(text_list),100):
        
#        print text
        if 0 in length_list[ind:min(ind+100,len(text_list))]:
            print ind
        seq,temp_score = run(text_list[ind:min(ind+100,len(text_list))],dict_list[ind:min(ind+100,len(text_list))],length_list[ind:min(ind+100,len(text_list))],sess)
        result_list.extend(seq)
        score_list.extend(temp_score)
    for item, item2 in zip(sample_text,result_list): 
        seg_text = visual(item,item2)             
#        print (seg_text)
        file.writelines(seg_text+'\n')
        result.append(seg_text)
    b = datetime.datetime.now()
    c = b-a
    print c.seconds
    file.close()
    df = pd.DataFrame(result)
    df.to_csv('result_CWS.csv')    
    np.save('cws_scores.npy',score_list)
'''    
    for cur_epoch in range(epoch):    
        for i in range(len(train_data_batches)):
            tf_unary_scores, tf_transition_params, _ = sess.run([unary_scores, transition_params, train_op],feed_dict={x_t:train_data_batches[i][0],y_t:train_data_batches[i][1],sequence_lengths_t:train_data_batches[i][2]})
                        
           
            if i % 100 == 0:
                correct_labels = 0
                total_labels = 0
                for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, train_data_batches[i][1],train_data_batches[i][2]):
              # Remove padding from the scores and tag sequence.
                    tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                    y_ = y_[:sequence_length_]
                    
              # Compute the highest scoring sequence.
                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params)
              # Evaluate word-level accuracy.
#                    print viterbi_sequence
                    correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                    total_labels += sequence_length_
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Accuracy: %.2f%%" % accuracy)
#        file = open('prediction.txt','w')
            correct_labels = 0
            total_labels = 0
        for i in range(len(valid_data_batches)):
            tf_unary_scores, tf_transition_params = sess.run([unary_scores, transition_params],feed_dict={x_t:valid_data_batches[i][0],y_t:valid_data_batches[i][1],sequence_lengths_t:valid_data_batches[i][2]})
            for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, valid_data_batches[i][1],valid_data_batches[i][2]):
          # Remove padding from the scores and tag sequence.
                tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                y_ = y_[:sequence_length_]

          # Compute the highest scoring sequence.
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params)
          # Evaluate word-level accuracy.
                correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                total_labels += sequence_length_
        accuracy = 100.0 * correct_labels / float(total_labels)
        print("Valid Accuracy: %.2f%%" % accuracy)
        train_data_batches = batch_maker(train_data,200)
        saver.save(sess,  'model/' + 'model_full_sentences_LSTM_CRF_punc_cut_4cls' + str(cur_epoch) + '.ckpt')
####################Core!! 
'''