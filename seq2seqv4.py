import tensorflow as tf
import helpers
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
sess = tf.InteractiveSession()

class HyperParameters():
    def __init__(self):
        self.encoder_hidden_units = 20
        self.decoder_hidden_units = 20
        self.vocab_size = 10
        self.input_embedding_size = 20

PAD = 0
EOS = 1

class Seq2SeqModel():
    def __init__(self, hparams):
        self.vocab_size = hparams.vocab_size
        self.input_embedding_size = hparams.input_embedding_size
        self.encoder_hidden_units = hparams.encoder_hidden_units
        self.decoder_hidden_units = hparams.decoder_hidden_units

        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')


    def build_graph(self, scope=None):
        dtype = tf.float32

        # set embeddings:
        embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), dtype=dtype)

        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)

        with tf.variable_scope( scope or "dynamic_seq2seq", dtype=dtype):
            #Encoder
            encoder_outputs, encoder_state = self.build_encoder(encoder_inputs_embedded)

            #Decoder
            decoder_logits, decoder_prediction = self.build_decoder(encoder_state, decoder_inputs_embedded)

            self.decoder_prediction = decoder_prediction
            self.decoder_logits = decoder_logits

            #Training
            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=dtype),
                logits=self.decoder_logits,
            )

            self.loss = tf.reduce_mean(stepwise_cross_entropy)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

            return self.loss, self.train_op, self.decoder_prediction

    def initialization(self, sess):
        sess.run(tf.global_variables_initializer())

    def update(self, sess, train_op, loss, fd):
        return sess.run([train_op, loss], fd)

    def calculate_loss(self, sess, loss, fd):
        return sess.run(loss, fd)

    def predict(self, sess, decoder_prediction, fd):
        return sess.run(decoder_prediction, fd)

    def build_encoder(self, encoder_inputs_embedded):
        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype

            cell = self.build_encoder_cell()
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell,
                encoder_inputs_embedded,
                dtype=dtype,
                time_major=True
            )

        return encoder_outputs, encoder_state

    def build_encoder_cell(self):
        return tf.contrib.rnn.LSTMCell(self.encoder_hidden_units)

    def build_decoder(self, encoder_state, decoder_inputs_embedded):
        cell = self.build_decoder_cell()

        with tf.variable_scope("decoder") as scope:
            dtype = scope.dtype

            decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                cell,
                decoder_inputs_embedded,
                initial_state = encoder_state,
                dtype=dtype,
                time_major=True
            )

            decoder_logits = tf.contrib.layers.linear(decoder_outputs, self.vocab_size)
            decoder_prediction = tf.argmax(decoder_logits, 2)

            return decoder_logits, decoder_prediction

    def build_decoder_cell(self):
        return tf.contrib.rnn.LSTMCell(self.decoder_hidden_units)



if __name__=="__main__":
    hparams = HyperParameters()
    seq2seq_model = Seq2SeqModel(hparams)
    loss, train_op, decoder_prediction = seq2seq_model.build_graph()

    seq2seq_model.initialization(sess)


    batch_size = 100

    batches = helpers.random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

    print('head of the batch')
    for seq in next(batches)[:10]:
        print(seq)

    def next_feed():
        batch = next(batches)
        encoder_inputs_, _ = helpers.batch(batch)
        decoder_targets_, _ = helpers.batch(
            [(sequence) + [EOS] for sequence in batch ]
        )
        decoder_inputs_, _ = helpers.batch(
            [ [EOS] + (sequence) for sequence in batch ]
        )
        return {
            seq2seq_model.encoder_inputs: encoder_inputs_,
            seq2seq_model.decoder_inputs: decoder_inputs_,
            seq2seq_model.decoder_targets: decoder_targets_,
        }

    loss_track = []

    max_batches = 3001
    batches_in_epoch = 1000

    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = seq2seq_model.update(sess, train_op, loss, fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('    minibatch loss: {}'.format(seq2seq_model.calculate_loss(sess, loss, fd)))
                predict_ = seq2seq_model.predict(sess, decoder_prediction, fd)

                for i, (inp, pred) in enumerate(zip(fd[seq2seq_model.encoder_inputs].T, predict_.T)):
                    print('   sample {}:'.format(i + 1))
                    print('      input    > {}'.format(inp))
                    print('      predicted > {}'.format(pred))
                    if i >= 2:
                        break

                print()

    except KeyboardInterrupt:
        print('trainning interrupted')


    plt.plot(loss_track)
    plt.show()
    print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))


