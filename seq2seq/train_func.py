# -*- coding:utf-8 -*-

import time
import tensorflow as tf
from func.seq_helper import *

def train_model(model, vocab, params, checkpoint_manager):
    epochs = params['epochs']
    batch_size = params['batch_size']
    
    pad_index = vocab.word2id[vocab.PAD_TOKEN]
    start_index = vocab.word2id[vocab.START_DECODING]
    
    params["vocab_size"] = vocab.count

    optimizer = tf.keras.opftimizers.Adam(name='Adam', learning_rate=0.01)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, pad_index))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    # @tf.function
    def train_step(encode_input, decode_target):
        batch_loss = 0
        with tf.GradientTape() as tape:
            encode_output, encode_hidden = model.call_encoder(encode_input)
            # 第一个decode输入开始标签
            decode_input = tf.expand_dims([start_index] * batch_size, 1)
            # 第一个隐藏层的输入
            decode_hidden = encode_hidden
            # 逐个预测序列
            predictions, _ = model(decode_input, decode_hidden, encode_output, decode_target)

            batch_loss = loss_function(decode_target[:, 1:], predictions)

            variables = model.encoder.trainable_variables + model.decode.trainable_variables + model.attention.trainable_variables

            gradients = tape.gradient(batch_loss, variables)
            
            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

    dataset, steps_per_epoch = train_batch_generator(batch_size)
    
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))


        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
    
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

                                                    
   
            


    