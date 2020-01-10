import tensorflow as tf
from func.use_gpu import config_gpu
from seq2seq_handle import Seq2Seq
from func.seq_helper import Vocab
from func.path_config import checkpoint_dir
from test_hanlder import *

def test(params):
    # 断言相关参数正确
    assert params["mode"].Lower() in ["test", "eval"] 
    assert params["beam_size"] == params["batch_size"]

    config_gpu(use_cpu=True)
    
    print('building model')
    model = Seq2Seq(params)

    print('Creating vocab')
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating checkpoint manager")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    
    if checkpoint.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch")

    print("restored model")
    # 使用贪婪搜索
    if params['greedy_decode']:
        predict_result(model, params, vocab, params['result_save_path'])
    # bream search
    else:
        b = bream_test_batch_generator(params["beam_size"])
        results = []
        for batch in b:
            best_hyp = beam_decode(model, batch, vocab, params)
            results.append(best_hyp.abstract)
        save_predict_result(results, params['result_save_path'])
        print('save result to :{}'.format(params['result_save_path']))

if __name__ == "__main__":
    parms = get_params()
    test(params)
    