# -*- coding:utf-8 -*-

import jieba
import pandas as pd
import numpy as np
import re
from functools import partial
from multiprocessing import cpu_count, Pool
from path_config import *

def read_data(train_data_path, test_data_path):
	return pd.read_csv(train_data_path), pd.read_csv(test_data_path)

def empty_value_handle(train_data, test_data):
	return train_data.dropna(subset=\
		['Question', 'Dialogue', 'Report']),\
		test_data.dropna(subset=['Question', 'Dialogue'])

def acquire_stop_word(path):
	with open(path, 'r', encoding='utf-8') as f:
		stop_words = f.readlines()
		stop_word_list = [word.strip() for word in stop_words]
		return stop_word_list

def clean_sentence(sentence):
	if isinstance(sentence, str):
		return re.sub(
            r'[\s+\-\!\/\|\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            '', sentence)
	else:
		return ' '

def filter_stop_word(words, stop_words):
	return [word for word in words if word not in stop_words]

def sentence_concat_process(sentence, stop_words):
	# 清除无用字符
	sententce = clean_sentence(sentence)
	# 切词
	words = jieba.cut(sententce)
	# 过滤停用词
	words = filter_stop_word(words, stop_words)
	return ' '.join(words)

def data_processing(df, stop_words):
	for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
		df[col_name] = df[col_name].apply(sentence_concat_process, args=(stop_words,))
	if "Report" in df.columns:
		df['Report'] = df['Report'].apply(sentence_concat_process, args=(stop_words,))
	return df

def parallelize(df, func, stop_words):
	cores = cpu_count()
	data_split = np.array_split(df, cores)
	with Pool(processes=cores) as pool:
		data = pd.concat(pool.map(partial(func,stop_words=stop_words), data_split))
	return data

def merge_data(train_df, test_df):
	train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
	test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
	merged_df = pd.concat([train_df['merged'], test_df['merged']], axis=0)
	return merged_df

def save_df(train_df, test_df, merged_df, train_path, test_path, merged_path):
	train_df = train_df.drop(['merged'], axis=1)
	test_df = test_df.drop(['merged'], axis=1)
	train_df.to_csv(train_path, index=None, header=True, encoding='utf-8')
	test_df.to_csv(test_path, index=None, header=True, encoding='utf-8')
	merged_df.to_csv(merged_path, index=None, header=True, encoding='utf-8')

if __name__ == '__main__':
	jieba.load_userdict(user_dict)
	train_df, test_df = read_data(train_data_path, test_data_path)
	train_df, test_df = empty_value_handle(train_df, test_df)
	stop_words = acquire_stop_word(stop_word_path)	
	train_df = parallelize(train_df, data_processing, stop_words)
	test_df = parallelize(test_df, data_processing, stop_words)
	merged_df = merge_data(train_df, test_df)
	save_df(
		train_df,test_df, merged_df, train_seg_path, test_data_path, merger_seg_path
		)