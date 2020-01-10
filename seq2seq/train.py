# -*- coding:utf-8 -*-

import tensorflow as tf
from func.use_gpu import config_gpu
from func.seq_helper import *
from seq2seq_handle import Seq2Seq
from train_func import train_model
from func.get_params import get_params


def train(params):
    # GPU config
    config_gpu(use_cpu=True)

    # load vocab
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    
    params['vocab_size'] = vocab.count


    print('building the model..')
    model = Seq2Seq(params)

    # 保存cheackpoint
    cheackpoint = tf.train.Checkpoint(Seq2Seq=model)
    cheackpoint_manager = tf.train.CheckpointManager(cheackpoint, params['checkpoint_dir'], max_to_keep = 5)

    train_model(model, vocab, params, cheackpoint_manager)

    
if __name__ == "__main__":
    params = get_params()
    train(params)