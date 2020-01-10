# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
from path_config import embedding_matrix_path, train_x_path, train_y_path, test_x_path

# load embedding
def load_embedding_matrix():
    return np.load(embedding_matrix_path + '.npy')

def load_train_Dataset(max_enc_len, max_dec_len):
    train_X = np.load(train_x_path + '.npy')
    train_Y = np.load(train_y_path + '.npy')
    
    train_X = train_X[:, :max_enc_len] 
    train_Y = train_Y[:, :max_dec_len]
    return train_X, train_Y

def load_test_dataset(max_enc_len):
    test_X = np.load(test_x_path)[:, :max_enc_len]
    return test_X

def train_batch_generator(batch_size, max_enc_len=200, max_dec_len=150):
    train_X, train_Y = load_train_Dataset(max_enc_len, max_dec_len)
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    steps_per_epoch = len(train_X) // batch_size

    return dataset, steps_per_epoch

def greedy_decode(model, test_X, params, vocab):
    batch_size = params['batch_size']
    # 返回的结果list
    results = []
    # 输入的样本数
    sample_size = len(test_X)
    # batch 操作轮数 math.ceil向上取整 小数 +1
    steps_epoch = math.ceil(sample_size / batch_size)
    for i in tqdm(range(steps_epoch)):
        batch_data = test_X[i * batch_size:(i+1) * batch_size]
        results += batch_greedy_decode(model, batch_data, vocab, params)        
    return results    

def batch_greedy_decode(model, batch_data, vocab, params):
    batch_size = len(batch_data)
    predicts = [''] * batch_size

    inps = tf.convert_to_tensor(batch_data)
    # 初始化隐藏层的输入
    hidden = [tf.zeros((batch_size, params['enc_units']))]
    # cerate encoder
    enc_output, enc_hidden = model.encoder(inps, hidden)
    dec_hidden = enc_hidden
    # <START>*BATCH_SIZE
    dec_input = tf.expand_dims([vocab.word_to_id(vocab.STOP_DECONDING)] * batch_size,1)
    
    context_vector, _ = model.attention(dec_hidden, enc_output)

    for t in range(params['max_dec_len']):
        # 上下文计算
        context_vector, attention_weights = model.attention(dec_hidden, enc_output)
        predictions, dec_hidden = model.decoder(dec_input,
                                                dec_hidden,
                                                enc_output,
                                                context_vector)
        # id转换，贪婪搜索
        predicted_ids = tf.argmax(predictions, axis=1).numpy()

        for index_, predicter_id in enumerate(predicted_ids):
            predicts[index_] += vocab.id_to_word(predicter_id) + ' '
        
        # 使用predicted_ids dim + 1 , 本次更新的dec_hidden 作为下一个词的预测输入
        dec_input = tf.expand_dims(predicter_id, 1)

    results = []
    for predict in predicts:
        predict = predict.strip()
        if vocab.STOP_DECONDING in predict:
            # 截断结束标记前的内容
            predict = predict[:predict.index(vocab.STOP_DECODING)]
        results.append(predict)
    return results


class Vocab:
    PAD_TOKEN = '<PAD>'
    UNKOWN_TOKEN = '<UNK>'
    START_DECODING = '<START>'
    STOP_DECONDING = '<STOP>'

    def __init__(self, vocab_file, vocab_max_size=None):
        """
        Vocab 对象,vocab基本操作封装
        :param vocab_file: Vocab 存储路径
        :param vocab_max_size: 最大字典数量
        """
        self.word2id, self.id2word = self.load_vocab(vocab_file, vocab_max_size)
        self.count = len(self.word2id)

    @staticmethod
    def load_vocab(file_path, vocab_max_size=None):
        vocab = {} 
        reverse_vocab = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word, index = line.strip().split("\t")
                index = int(index)
                # 如果vocab 超出指定大小，　跳出循环并截断
                if vocab_max_size and index > vocab_max_size:
                    break
                    
                vocab[word] = index
                reverse_vocab[index] = word
        return vocab, reverse_vocab


    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id) 
        return self.id2word[word_id]

    
