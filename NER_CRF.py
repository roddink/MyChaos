# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:45:28 2017

@author: sz_rafahao
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from collections import Counter
from langconv import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tag_token = {'O':0,'B-company_name':1,'I-company_name':2,'E-company_name':3,'S-company_name':4,'B-time':5,'I-time':6,'E-time':7,'S-time':8,'B-job_title':9,'I-job_title':10,'E-job_title':11,'S-job_title':12,'B-person_name': 13,'I-person_name': 14,'E-person_name': 15,'S-person_name': 16,'B-location':17,'I-location':18,'E-location':19,'S-location':20}
#file = pd.read_csv('C:/NER/news_ner/boson_ner.all.csv')
#text = file.values[:,-1].tolist()
#text_sep = []
#for item in text:
#    if not(item is np.nan):
#        text_sep.extend(item.split('。/O '))
#file_write = open('C:/NER/news_ner/boson_ner.txt','w')
#for item in text_sep:
#    file_write.writelines(item+'\n')
#file_write.close()

def parse(file_name):
    file_NER = open(file_name)
    text = file_NER.readlines()
    text_input = []
    label_input = []
    length_input = []
    for ind in range(len(text)):
        if text[ind] != '':
            temp_sentence = []
            temp_sentence_label = []
            text[ind] = text[ind].strip().split(' ')
            for ind2 in range(len(text[ind])):
                temp = text[ind][ind2].split('/')
                if len(temp) == 2:
                    temp_sentence.append(temp[0])
                    temp_sentence_label.append(temp[1])
            length_input.append(len(text[ind]))
            text_input.append(temp_sentence)
            label_input.append(temp_sentence_label)
    return text_input,label_input,length_input


def tokenizer(text,token,max_len):# map text to token, INPUT: text, token dictionary, max_len
    text_new = []
    for i in range(len(text)):
        temp = np.zeros(max_len,dtype=int)
        for j in range(min(len(text[i]),max_len)):
            temp[j] = token.get(text[i][j],0)
        text_new.append(temp.tolist())
    return text_new

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


def init_embedding(gensim_model,embedding_size): # tra2sim embedding initialization, INPUT: gensim model, embedding_size of gensim model
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

def batchmaker(text_input,text_bi,text_tri,label_input,length_input,batch_size=100):
    batch = []
    for ind in range(0,len(text_input),batch_size):
        batch.append([text_input[ind:ind+batch_size],text_bi[ind:ind+batch_size],text_tri[ind:ind+batch_size],label_input[ind:ind+batch_size],length_input[ind:ind+batch_size]])
    return batch[:-1]

class NER_model():
    def __init__(self,n_tags=21,n_hidden=100,max_len=500,batch_size=100,embedding_size=300,vocab_s = 5000):
        self.n_hidden = n_hidden
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_tags = n_tags
        self.embedding_size = embedding_size
        self.vocab_s = vocab_s
        self.graph = self.graph_def()
    def graph_def(self):
        with tf.Graph().as_default() as graph:
            with tf.variable_scope('input_output'):
                self.input_x0 = tf.placeholder(tf.int32, [None, None],name='text_t0')
                self.input_x1 = tf.placeholder(tf.int32, [None, None],name='text_t1')
                self.input_1x = tf.placeholder(tf.int32, [None, None],name='text_1t')
                self.label = tf.placeholder(tf.int32,[None,None],name='label')
                self.sequence_length = tf.placeholder(tf.int32,[None],name='length')
                self.keep_prob = tf.placeholder(tf.float32,[1],name='drop_out_ratio')
            with tf.variable_scope('embeddings'):
                self.uni_embedding = tf.get_variable("embedding_unigram", [self.vocab_s, self.embedding_size], trainable=True)
                self.bi_embedding = tf.get_variable("embedding_bigram", [self.vocab_s, self.embedding_size], trainable=True)
                self.tri_embedding = tf.get_variable("embedding_trigram", [self.vocab_s, self.embedding_size], trainable=True)
            with tf.variable_scope('lookups'):
                uni_embedding_input_x0 = tf.nn.embedding_lookup(self.uni_embedding, self.input_x0)
                bi_embedding_input_x0 = tf.nn.embedding_lookup(self.bi_embedding,self.input_x0)
                tri_embedding_input_x0 = tf.nn.embedding_lookup(self.tri_embedding,self.input_x0)
                bi_embedding_input_x1 = tf.nn.embedding_lookup(self.bi_embedding,self.input_x1)
                tri_embedding_input_x1 = tf.nn.embedding_lookup(self.tri_embedding,self.input_x1)
                tri_embedding_input_1x = tf.nn.embedding_lookup(self.tri_embedding,self.input_1x)
                concate_input = tf.concat([uni_embedding_input_x0,bi_embedding_input_x0,tri_embedding_input_x0,bi_embedding_input_x1,tri_embedding_input_x1,tri_embedding_input_1x],-1)
            with tf.variable_scope('Bi-GRU'):
                cell_fw = tf.contrib.rnn.GRUCell(self.n_hidden)
                cell_bw = tf.contrib.rnn.GRUCell(self.n_hidden)
                _initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
                _initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob[0])
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob[0])
                outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, concate_input, time_major=False, parallel_iterations=100,sequence_length=self.sequence_length, initial_state_fw=_initial_state_fw,initial_state_bw=_initial_state_bw,scope='GRU')
                outputs = tf.concat(outputs, 2)
            with tf.variable_scope('CRF'):
                weights = tf.get_variable("weights_CRF", [2*self.n_hidden, self.n_tags]) 
                self.transition_params_CRF = tf.get_variable("trans_mat_cws", [self.n_tags, self.n_tags]) 
#                outputs_cws_share = tf.concat([outputs,outputs_share,x_dict_t], -1)                         
                matricized_x_t_cws = tf.reshape(outputs, [-1, 2*self.n_hidden])
                matricized_unary_scores = tf.matmul(matricized_x_t_cws, weights)
                self.unary_scores = tf.reshape(matricized_unary_scores,[self.batch_size, self.max_len, self.n_tags])     
                log_likelihood, transition_params_cws = tf.contrib.crf.crf_log_likelihood(self.unary_scores, self.label, self.sequence_length,transition_params=self.transition_params_CRF)
            self.loss = tf.reduce_mean(-log_likelihood)
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
            self.init = tf.global_variables_initializer()#initize all variables
        return graph
    
    def train(self,init_embedding,train_batch,valid_batch,epoch=100):
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init)
            writer = tf.summary.FileWriter('tensorboard_log', sess.graph)
            saver = tf.train.Saver(max_to_keep=10)
            if not (init_embedding is None):
                sess.run(tf.assign(self.uni_embedding,init_embedding))
                sess.run(tf.assign(self.bi_embedding,init_embedding))
                sess.run(tf.assign(self.tri_embedding,init_embedding))
            for cur_epoch in range(epoch):
                print 'epoch: '+str(cur_epoch)
                for i in range(len(train_batch)):
                    tf_unary_scores_, tf_transition_params_,_ = sess.run([ self.unary_scores, self.transition_params_CRF, self.train_op],feed_dict={self.input_x0:train_batch[i][0],self.input_x1:train_batch[i][1],self.input_1x:train_batch[i][2],self.label:train_batch[i][3],self.sequence_length:train_batch[i][4],self.keep_prob:[0.8]})
                    if i % 30 == 1:
                        correct_labels = 0
        
                        total_labels = 0
                        for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores_, train_batch[i][3],train_batch[i][4]):
                      # Remove padding from the scores and tag sequence.
                            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                            y_ = y_[:sequence_length_]
                            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params_)
                            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                            total_labels += sequence_length_
                        accuracy = 100.0 * correct_labels / float(total_labels)
                        print("Train_Accuracy: %.2f%%" % accuracy)
                correct_labels = 0
                total_labels = 0
                for i in range(len(valid_batch)):
                    tf_unary_scores_, tf_transition_params_ = sess.run([ self.unary_scores, self.transition_params_CRF],feed_dict={self.input_x0:valid_batch[i][0],self.input_x1:valid_batch[i][1],self.input_1x:valid_batch[i][2],self.label:valid_batch[i][3],self.sequence_length:valid_batch[i][4],self.keep_prob:[1.0]})
                    for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores_, valid_batch[i][3],valid_batch[i][4]):
                  # Remove padding from the scores and tag sequence.
                        tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                        y_ = y_[:sequence_length_]
                        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params_)
                        correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                        total_labels += sequence_length_
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Valid_Accuracy: %.2f%%" % accuracy)
                saver.save(sess,  'model/' + 'NER'+str(self.n_hidden)+'_' + str(accuracy) + '.ckpt')
                
    def predict(self,predict_batch,model_path):
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init)
            writer = tf.summary.FileWriter('tensorboard_log', sess.graph)
            saver = tf.train.Saver()
            saver.restore(sess,model_path)
            tf_unary_scores_, tf_transition_params_ = sess.run([ self.unary_scores, self.transition_params_CRF],feed_dict={self.input_x0:predict_batch[0],self.input_x1:predict_batch[1],self.input_1x:predict_batch[2],self.sequence_length:predict_batch[4],self.keep_prob:[1.0]})
            predict_sequences = []
            true_sequences = []
            for tf_unary_scores_, y_,sequence_length_ in zip(tf_unary_scores_,predict_batch[3],predict_batch[4]):
                 tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                 y_ = y_[:sequence_length_]
                 viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params_)
                 predict_sequences.append(viterbi_sequence)
                 true_sequences.append(y_)
        return predict_sequences,true_sequences
if __name__=="__main__":
    max_len = 500
    embedding_size = 300
    tag_token = {'O':0,'B-company_name':1,'I-company_name':2,'E-company_name':3,'S-company_name':4,'B-time':5,'I-time':6,'E-time':7,'S-time':8,'B-job_title':9,'I-job_title':10,'E-job_title':11,'S-job_title':12,'B-person_name': 13,'I-person_name': 14,'E-person_name': 15,'S-person_name': 16,'B-location':17,'I-location':18,'E-location':19,'S-location':20,'B-org_name':21,'I-org_name':22,'E-org_name':23,'S-org_name':24,'B-product_name':25,'I-product_name':26,'E-product_name':27,"S-product_name":28}
    file = pd.read_csv('boson_ner.all.csv')
    text = file.values[:,-1].tolist()
    file2 = pd.read_csv('boson_ner.finance.csv')
    text += file2.values[:,-1].tolist()
    text_sep = []
    for item in text:
        if not(item is np.nan):
            text_sep.extend(item.split('。/O '))
    file_write = open('boson_ner.txt','w')
    for item in text_sep:
        file_write.writelines(item+'\n')
    file_write.close()
    text_input, label_input, length_input = parse('boson_ner.txt')
    gensim_model = Word2Vec.load('data/mongoDB_w2v_epoch5.txt')
    embedding, dicts, _ = init_embedding(gensim_model,embedding_size)
    text_input = tokenizer(text_input,dicts,max_len)
    label_input = tokenizer(label_input,tag_token,max_len)
    length_over = np.where(np.array(length_input)>500)[0]
    for item in length_over:
        length_input[item] = 500
    text_bi = bigram_maker(text_input)
    text_tri = trigram_maker(text_input)
    train_batch = batchmaker(text_input[:30000]+text_input[50000:],text_bi[:30000]+text_bi[50000:],text_tri[:30000]+text_tri[50000:],label_input[:30000]+label_input[50000:],length_input[:30000]+length_input[50000:],batch_size=100)
    valid_batch = batchmaker(text_input[30000:50000],text_bi[30000:50000],text_tri[30000:50000],label_input[30000:50000],length_input[30000:50000],batch_size=100)
    ner = NER_model(vocab_s = len(dicts)+1)
    ner.train(embedding,train_batch,valid_batch)     
#    haha,hehe = ner.predict(valid_batch[-1],'model/NER100_93.3839894656.ckpt')
#    np.save('NER_pred.npy',haha)
#    np.save('NER_true.npy',hehe)
    