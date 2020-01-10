import tensorflow as tf
from func.seq_helper import load_embedding_matrix
from seq2seq_model import *

class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        # load_preprocessed_embedding
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        # Encoder 
        self.encoder = Encoder(params["vocab_size"], 
                               params["embed_size"],
                               self.embedding_matrix,
                               params["enc_units"],
                               params["batch_size"])

        self.attention = BahdanauAttention(params["attn_units"])

        self.decoder = Decoder(params["vocab_size"],
                               params["embed_size"],
                               self.embedding_matrix,
                               params["dec_units"],
                               params['batch_size'])

        def call_encode(self, enc_inp):
            enc_hidden = self.encoder.initialize_hidden_state()
            enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
            return enc_output, enc_hidden
        
        def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
            # dec_hidden = enc_hidden
            context_vector, attention_weights = self.attention(dec_hidden, enc_output)
            
            pred, dec_hidden = self.decoder(dec_input,
                                            None, 
                                            None,
                                            context_vector)

            return pred, dec_hidden, context_vector, attention_weights

        def call(self, dec_input, dec_hidden, enc_output, dec_target):
            predictions = []
            attentions = []

            context_vector, _ = self.attention(dec_hidden, enc_output)

            for t in range(1, dec_target.shape[1]):
                pred, dec_hidden = self.decoder(dec_input, 
                                                dec_hidden,
                                                enc_output,
                                                context_vector)

                context_vector, attn = self.attention(dec_hidden, enc_output)
                # use teacher forcing
                dec_input = tf.expand_dims(dec_target[:t], 1)

                predictions.append(pred)
                attentions.append(attn)
            
            return tf.stack(predictions, 1), dec_hidden
                

