# -*- coding:utf-8 -*-

from path_config import merger_seg_path, embedding_dim, save_wv_model_path, vocab_path
from gensim.models import word2vec

if __name__ == '__main__':
	# 1.训练词向量
	print('training_word2vec')
	print(merger_seg_path)
	model_wv = word2vec.Word2Vec(word2vec.LineSentence(merger_seg_path), sg=1, workers=8, min_count=5, size=300)
	print(word2vec)
	# 2.保存模型, vocab
	vocab = {word:index for index, word in enumerate(model_wv.wv.index2word)}	
	with open(vocab_path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(vocab))
	model_wv.save(save_wv_model_path)
	# 3.读取模型
	word2vec.Word2Vec.load(save_wv_model_path)
	# 4.生成embedding_matrix
	embedding_matix = model_wv.wv.vectors
	print(embedding_matix.shape)

