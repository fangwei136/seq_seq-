# -*- coding:utf-8 -*-

import json
import pandas as ps
import numpy as np
from path_config import *
from gensim.models import word2vec

def get_max_len(df_x):
	max_lens = df_x.apply(lambda x: x.count(' ') + 1)
	return int(np.mean(max_lens) + 2 * np.std(max_lens))

def pad_process(sentence, maxlen, vocab):
	words = sentence.strip().split(' ')
	words_clip = words[:maxlen]
	sentence = [word if word in vocab else '<UNK>' for word in words_clip]
	sentence = sentence + ['<PAD>'] * (maxlen - len(words_clip))
	sentence = ['<START>'] + sentence + ['<STOP>']
	return ' '.join(sentence)

def save_dict(path, vocab):
	with open(path, 'w', encoding='utf-8') as f:
		f.write(json.dumps(vocab))

def transfrom_data(sentence, vocab):
	words = sentence.split(' ')
	index_ = [vocab[word] for word in words]
	return index_

if __name__ == '__main__':
	model_wv = word2vec.Word2Vec.load(save_model_path)
	vocab = model_wv.wv.vocab
	# print(vocab)
	train_df = ps.read_csv(train_processed_path, encoding='utf-8')
	test_df = ps.read_csv(test_processed_path, encoding='utf-8')
	print(train_df.info())
	# 分离数据
	train_df['x'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join((str(i) for i in x)), axis=1)
	test_df['x'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join((str(i) for i in x)), axis=1)
	train_df['y']= train_df[['Report']].apply(lambda x: ' '.join((str(i) for i in x)), axis=1)
	# 填充符号
	# 获取最大长度
	train_x_max_len = get_max_len(train_df['x'])
	test_x_max_len = get_max_len(test_df['x'])
	print(f'train_len({train_x_max_len}), test_len({test_x_max_len})')
	x_max_len = max(train_x_max_len, test_x_max_len)
	train_df['X'] = train_df['x'].apply(lambda x: pad_process(x, x_max_len, vocab))
	test_df['X'] = train_df['x'].apply(lambda x:pad_process(x, x_max_len, vocab))

	train_y_max_len = get_max_len(train_df['y'])
	train_df["Y"] = train_df['y'].apply(lambda x: pad_process(x, train_y_max_len, vocab))

	train_df['X'].to_csv(train_x_path, index=None, header=False, encoding='utf-8')
	train_df['Y'].to_csv(train_y_path, index=None, header=False, encoding='utf-8')
	test_df['X'].to_csv(test_x_path, index=None, header=False, encoding='utf-8')
	# retrain w2v model
	model_wv.build_vocab(word2vec.LineSentence(train_x_path), update=True)
	model_wv.train(word2vec.LineSentence(train_x_path), epochs=wv_train_epochs, total_examples=model_wv.corpus_count)

	model_wv.build_vocab(word2vec.LineSentence(train_y_path), update=True)
	model_wv.train(word2vec.LineSentence(train_y_path), epochs=wv_train_epochs, total_examples=model_wv.corpus_count)

	model_wv.build_vocab(word2vec.LineSentence(test_x_path), update=True)
	model_wv.train(word2vec.LineSentence(test_x_path), epochs=wv_train_epochs, total_examples=model_wv.corpus_count)

	model_wv.save(save_model_path)

	vocab = {word: index_ for index_, word in enumerate(model_wv.wv.index2word)}
	reverse_vocab = {index_: word for index_, word in enumerate(model_wv.wv.index2word)}

	save_dict(vocab_path, vocab)
	save_dict(reverse_vocab_path, reverse_vocab)
	
	# 保存矩阵
	embedding_matrix = model_wv.wv.vectors
	np.savetxt(embedding_matrix_path, embedding_matrix, fmt='%0.8f')

	# 将数据集词转换为索引
	train_index_x = train_df['X'].apply(lambda x: transfrom_data(x, vocab))
	train_index_y = train_df['Y'].apply(lambda x: transfrom_data(x, vocab))
	test_index_x = test_df['X'].apply(lambda x: transfrom_data(x, vocab))

	# 数据转换为numpy数组
	train_X = np.array(train_index_x.tolist())
	train_Y = np.array(train_index_y.tolist())
	test_X = np.array(test_index_x.tolist())

	# 保存数据
	np.savetxt(train_arr_x_path, train_X, fmt='%0.8f')
	np.savetxt(train_arr_y_path, train_Y, fmt='%0.8f')
	np.savetxt(test_arr_x_path, test_X, fmt='%0.8f')

