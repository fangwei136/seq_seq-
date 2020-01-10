# -*- coding:utf-8 -*-

import tensorflow as tf
import pandas as pd
from func.seq_helper import load_test_dataset, greedy_decode

def predict_result(model, params, vocab, result_save_path):
    test_X = load_test_dataset(params['max_enc_len'])
    results = greedy_decode(model, test_X, params, vocab)
    
    save_predict_result(results, result_save_path)
    
def save_predict_result(results, result_save_path):
    test_df = pd.read_csv(result_save_path)
    test_df['Prediction'] = results
    test_df = test_df[['QID', 'Prediction']]
    test_df.to_csv(result_save_path, index=None, sep=',')

def bream_test_batch_generator(bream_size, max_enc_len):
    test_X = load_test_dataset(max_enc_len)
    for row in test_X:
        bream_search_data = tf.convert_to_tensor([row for i in range(bream_size)])
        yield bream_search_data
