# from __future__ import division
# import numpy as np
# import tensorflow as tf
# import re
# from collections import Counter
# import operator
# from tensorflow.contrib import learn

# embeddingsz = 50
# batchsz = 100
# num_steps = 20
# keepprob = 0.5
# learning_rate = 1e4
# dropout = 0.5
# hiddensz = 256
# epochs = 5

# def basic_tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
#     """
#     Very basic tokenizer: split the sentence into a list of tokens, lowercase.
#     """
#     words = []
#     for space_separated_fragment in sentence.strip().split():
#         words.extend(re.split(word_split, space_separated_fragment))
#     return [w.lower() for w in words if w]

# def readfile(train_file):
#     asterisks = 0
#     startreading = False
#     words = []
#     with open(train_file, 'r') as training:
#         for line in training:
#             tokens = basic_tokenizer(line)
#             if startreading == False and asterisks < 2:
#                 for token in tokens:
#                     if token == "***":
#                         asterisks += 1
#                         if asterisks == 2:
#                             startreading = True
#             elif startreading:
#                 for i in range(len(tokens)):
#                     if tokens[i] == "***":
#                         # stop reading
#                         asterisks += 1
#                         startreading = False
#                         break
#                     if tokens[i] == ".":
#                         tokens.insert(i + 1, "STOP")
#                 if asterisks > 2:
#                     break
#                 words.extend(tokens)
#         c = Counter(words)
#         sorted_c = sorted(c.items(), key=operator.itemgetter(1), reverse=True)[:8000]
#         vocabulary = dict(zip([str(i[0]) for i in sorted_c], [integer for integer in range(8000)]))
#         vocabulary["*UNK*"] = 8000
#         for index in range(len(words)):
#             if vocabulary.has_key(words[index]) == False:
#                 words[index] = "*UNK*"
#         words_list = map(lambda x: vocabulary[x], words)
#         training_data, testing_data = words_list[:int(len(words_list) * .9)], words_list[int(len(words_list)*.9):]
#         return np.array(training_data[:-1]), np.array(training_data[1:]), np.array(testing_data[:-1]), np.array(testing_data[:-1]), 8001


# # class LSTM():
# #     def __init__(self, vocabsz):
# #
# #         self.X = tf.placeholder(tf.int32, [batchsz, num_steps])
# #         self.Y = tf.placeholder(tf.int32, [batchsz, num_steps])  # Shape [batchsz, num_steps]
# #         self.keep_prob = tf.placeholder(tf.float32)
# #
# #         self.E = tf.Variable(tf.truncated_normal([vocabsz, embeddingsz], stddev=0.1))
# #         self.W = tf.Variable(tf.truncated_normal([hiddensz, vocabsz], stddev=0.1))
# #         self.b = tf.Variable(tf.constant(0.1, shape=[vocabsz]))
# #
# #         self.lstm = tf.nn.rnn_cell.BasicLSTMCell(hiddensz)
# #         self.initial_state = self.lstm.zero_state(batchsz, tf.float32)
# #
# #         embd = tf.nn.embedding_lookup(self.E, self.X)  # Shape [vocabsz, embeddingsz, batchsz]
# #         embd_drop = tf.nn.dropout(embd, self.keep_prob)
# #
# #         cell_output, out_state = tf.nn.dynamic_rnn(self.lstm, embd_drop, initial_state=self.initial_state)
# #
# #         self.final_state = out_state
# #
# #         output = tf.reshape(cell_output, [batchsz*num_steps, hiddensz])
# #         self.logits = tf.matmul(output, self.W) + self.b
# #
# #         # Build the Loss Computation
# #         self.loss_val = self.loss()
# #
# #         # Build the Training Operation
# #         self.train_op = self.train()
# #
# #
# #     def loss(self):
# #         reshaped_y = tf.reshape(self.Y, [batchsz*num_steps])
# #         loss1 = tf.nn.seq2seq.sequence_loss_by_example(
# #             [self.logits],
# #             [reshaped_y],
# #             [tf.ones([batchsz*num_steps], tf.float32)])
# #         return tf.reduce_sum(loss1) / batchsz
# #
# #     def train(self):
# #         opt = tf.train.AdamOptimizer(learning_rate)
# #         return opt.minimize(self.loss_val)

# if __name__ == "__main__":
#     train_x, train_y, test_x, test_y, vocabsz= readfile('dracula.txt')
#     num_train = len(train_x)

#     # Launch Tensorflow Session
#     print 'Launching Session!'
#     with tf.Session() as sess:
#         # Instantiate Model
#         #lstm = LSTM(vocabsz)

#         X = tf.placeholder(tf.int32, [batchsz, num_steps])
#         Y = tf.placeholder(tf.int32, [batchsz, num_steps])  # Shape [batchsz, num_steps]
#         keep_prob = tf.placeholder(tf.float32)

#         E = tf.Variable(tf.truncated_normal([vocabsz, embeddingsz], stddev=0.1))
#         W = tf.Variable(tf.truncated_normal([hiddensz, vocabsz], stddev=0.1))
#         b = tf.Variable(tf.constant(0.1, shape=[vocabsz]))

#         lstm = tf.nn.rnn_cell.BasicLSTMCell(hiddensz, state_is_tuple=True)
#         initial_state = lstm.zero_state(batchsz, tf.float32)

#         embd = tf.nn.embedding_lookup(E, X)  # Shape [vocabsz, embeddingsz, batchsz]
#         embd_drop = tf.nn.dropout(embd, keep_prob)

#         cell_output, out_state = tf.nn.dynamic_rnn(lstm, embd_drop, initial_state=initial_state)

#         #final_state = out_state

#         output = tf.reshape(cell_output, [batchsz * num_steps, hiddensz])
#         logits = tf.matmul(output, W) + b

#         # Build the Loss Computation
#         reshaped_y = tf.reshape(Y, [batchsz * num_steps])
#         loss1 = tf.nn.seq2seq.sequence_loss_by_example(
#             [logits],
#             [reshaped_y],
#             [tf.ones([batchsz * num_steps], tf.float32)])
#         loss = tf.reduce_sum(loss1) / batchsz

#         # Build the Training Operation
#         train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#     #
#     # def loss(self):
#     #     reshaped_y = tf.reshape(self.Y, [batchsz * num_steps])
#     #     loss1 = tf.nn.seq2seq.sequence_loss_by_example(
#     #         [self.logits],
#     #         [reshaped_y],
#     #         [tf.ones([batchsz * num_steps], tf.float32)])
#     #     return tf.reduce_sum(loss1) / batchsz
#     #
#     #
#     # def train(self):
#     #     opt = tf.train.AdamOptimizer(learning_rate)
#     #     return opt.minimize(self.loss_val)


#         # Initialize all Variables
#         sess.run(tf.initialize_all_variables())

#         print 'Starting Training!'
#         loss_val, counter = 0.0, 0
#         num_batch = int((num_train - num_steps)) / batchsz
#         curr_state = [initial_state[0].eval(), initial_state[1].eval()]
#         print num_batch
#         #for start, end in zip(range(0, num_train, batchsz), range(batchsz, num_train + batchsz, batchsz)):
#         for epoch in range(epochs):
#             for i in range(int(num_batch)):
#                 x = []
#                 y = []
#                 #for i in range(num_steps):
#                 #    for j in range(start, end):
#                 for j in range(batchsz):
#                     x.append(train_x[i * batchsz + j: i * batchsz + j + num_steps])
#                     y.append(train_y[i * batchsz + j: i * batchsz + j + num_steps])
#                         #x.append(train_x[j])
#                         #y.append(train_y[j])
#                 x, y = np.array(x), np.array(y)
#                 #x.shape = (batchsz, num_steps)
#                 curr_loss, final_state, _ = sess.run([loss, out_state, train_op],
#                                         feed_dict={X: x,
#                                                    Y: y,
#                                                    keep_prob: keepprob,
#                                                    initial_state[0]: curr_state[0],
#                                                    initial_state[1]: curr_state[1]})
#                 curr_state = [final_state[0], final_state[1]]
#                 loss_val, counter = loss_val + curr_loss, counter + 1
#                 iters = counter * num_steps
#                 if counter % 100 == 0:
#                     print loss_val, iters
#                     print 'Batch {} Train Perplexity:'.format(counter), np.exp(curr_loss / iters)

#         test_loss, counter = 0.0, 0
#         curr_state = [initial_state[0].eval(), initial_state[1].eval()]
#         num_batch = int((len(test_x) - num_steps) / batchsz)
#         print len(test_x)
#         print num_batch
#         for i in range(num_batch):
#             x_test = []
#             y_test = []
#             for j in range(batchsz):
#                 x_test.append(test_x[i * batchsz + j: i * batchsz + j + num_steps])
#                 y_test.append(test_y[i * batchsz + j: i * batchsz + j + num_steps])
#             x_test, y_test = np.array(x_test), np.array(y_test)
#             loss_val, final_state = sess.run([loss, out_state], feed_dict={
#                 X: x_test,
#                 Y: y_test,
#                 keep_prob: 1,
#                 initial_state[0]: curr_state[0],
#                 initial_state[1]: curr_state[1]})
#             curr_state = [final_state[0], final_state[1]]
#             test_loss += loss_val
#             counter += 1
#             iters = counter * num_steps
#             if counter % 100:
#                 print 'Batch {} Test Perplexity:'.format(counter), np.exp(loss_val / iters)

#         #test_loss = sess.run(bigram.loss_val, feed_dict={bigram.X: test_x, bigram.Y: test_y})
#         print 'Test Perplexity:', np.exp(test_loss / iters)



"""
langmod_rnn.py

Object oriented version of Tensorflow code for the LSTM Language Model.
"""
from collections import defaultdict
import numpy as np
import re
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS

# Preprocessing Parameters
tf.app.flags.DEFINE_integer('max_vocab_size', 8000, 'Maximum Vocabulary Size.')

# Model Parameters
tf.app.flags.DEFINE_integer('num_steps', 20, 'Number of unrolled steps before backprop.')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Size of the Embeddings.')
tf.app.flags.DEFINE_integer('hidden_size', 256, 'Size of the LSTM Layer.')

# Training Parameters
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of Training Epochs.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Size of a batch (for training).')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for Adam Optimizer.')
tf.app.flags.DEFINE_float('dropout_prob', 0.5, 'Keep probability, for dropout.')
tf.app.flags.DEFINE_integer('eval_every', 10000, 'Print statistics every eval_every words.')

BOOK_PATH = "data/crime_punishment.txt"
STOP = "*STOP*"
UNK = "*UNK*"
UNK_ID = 0
STOP_ID = 1


class RNNLangmod():
    def __init__(self, vocab_size, embedding_size, num_steps, hidden_size, batch_size,
                 learning_rate):
        """
        Instantiate an RNNLangmod Model, with the necessary hyperparameters.

        :param vocab_size: Size of the vocabulary.
        :param num_steps: Number of words to feed into LSTM before performing a gradient update.
        :param hidden_size: Size of the LSTM Layer.
        :param num_layers: Number of stacked LSTM Layers in the model.
        :param batch_size: Batch size (for training).
        :param learning_rate: Learning rate for Adam Optimizer
        """
        self.vocab_size, self.embedding_size = vocab_size, embedding_size
        self.hidden, self.num_steps = hidden_size, num_steps
        self.bsz, self.learning_rate = batch_size, learning_rate

        # Setup Placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.num_steps])
        self.Y = tf.placeholder(tf.int32, shape=[None, self.num_steps])
        self.keep_prob = tf.placeholder(tf.float32)

        # Instantiate Network Weights
        self.instantiate_weights()

        # Build the Inference Graph
        self.logits, self.final_state = self.inference()

        # Build the Loss Computation
        self.loss_val = self.loss()

        # Build the Training Operation
        self.train_op = self.train()

    def instantiate_weights(self):
        """
        Instantiate the network Variables, for the Embedding, LSTM, and Output Layers.
        """
        # Embedding Matrix
        self.E = self.weight_variable([self.vocab_size, self.embedding_size], 'Embedding')

        # Basic LSTM Cell
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden)
        self.initial_state = self.cell.zero_state(self.bsz, tf.float32)

        # Softmax Output
        self.softmax_w = self.weight_variable([self.hidden, self.vocab_size], 'Softmax_Weight')
        self.softmax_b = self.weight_variable([self.vocab_size], 'Softmax_Bias')

    def inference(self):
        """
        Build the inference computation graph for the model, going from the input to the output
        logits (before final softmax activation).

        :return Tuple of 2D Logits Tensor [bsz * steps, vocab], and Final State [num_layers]
        """
        # Feed input through the Embedding Layer, Dropout.
        emb = tf.nn.embedding_lookup(self.E, self.X)                   # Shape [bsz, steps, hidden]
        drop_emb = tf.nn.dropout(emb, self.keep_prob)

        # Feed input through dynamic_rnn
        out, f_state = tf.nn.dynamic_rnn(self.cell, drop_emb,          # Shape [bsz, steps, hidden]
                                         initial_state=self.initial_state)

        # Reshape the outputs into a single 2D Tensor
        outputs = tf.reshape(out, [-1, self.hidden])                   # Shape [bsz * steps, hidden]

        # Feed through final layer, compute logits
        logits = tf.matmul(outputs, self.softmax_w) + self.softmax_b   # Shape [bsz * steps, vocab]
        return logits, f_state

    def loss(self):
        """
        Build the sequence cross-entropy loss by example computation.

        :return Scalar representing sequence loss.
        """
        seq_loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits],
                                                          [tf.reshape(self.Y, [-1])],
                                                          [tf.ones([self.bsz * self.num_steps])])
        loss = tf.reduce_sum(seq_loss) / self.bsz
        return loss

    def train(self):
        """
        Build the training operation, using the sequence loss by example and an Adam Optimizer.

        :return Training Operation
        """
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss_val)

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)


def basic_tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w]


def read():
    """
    Reads and parses the book specified by BOOK_PATH, and returns the vectorized representation
    of the text.

    :return: Triple consisting of the training inputs (x), training outputs (y), and the vocabulary.
    """
    # Tokenize and add raw tokens to a list.
    raw_tokens, vocabulary = [], defaultdict(int)
    with open(BOOK_PATH, 'r') as f:
        for line in f:
            words = basic_tokenizer(line)
            for w in words:
                vocabulary[w] += 1
            raw_tokens.extend([STOP] + words)

    # Create the vocabulary
    vocab_list = [UNK, STOP] + sorted(vocabulary, key=vocabulary.get, reverse=True)
    vocabulary = {vocab_list[i]: i for i in range(FLAGS.max_vocab_size) if i < len(vocab_list)}

    # Use the vocabulary to vectorize the data, return x, y, and vocabulary
    data = map(lambda tok: vocabulary.get(tok, UNK_ID), raw_tokens)
    train_len = int(len(data) * 0.9)
    return np.array(data[:train_len - 1], dtype=int), np.array(data[1:train_len], dtype=int), \
        np.array(data[train_len:-1], dtype=int), np.array(data[train_len + 1:], dtype=int), \
        vocabulary, vocab_list

# Main Training Block
if __name__ == "__main__":
    # Preprocess and vectorize the data
    x, y, test_x, test_y, vocab, vocab_list = read()

    # Launch Tensorflow Session
    print 'Launching Tensorflow Session'
    with tf.Session() as sess:
        # Instantiate Model
        rnn_langmod = RNNLangmod(len(vocab), FLAGS.embedding_size, FLAGS.num_steps,
                                 FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

        # Initialize all Variables
        sess.run(tf.initialize_all_variables())

        # Start Training
        ex_bsz, bsz, steps = FLAGS.batch_size * FLAGS.num_steps, FLAGS.batch_size, FLAGS.num_steps
        for epoch in range(FLAGS.num_epochs):
            state, loss, iters, start_time = sess.run(rnn_langmod.initial_state), 0., 0, time.time()
            for start, end in zip(range(0, len(x) - ex_bsz, ex_bsz), range(ex_bsz, len(x), ex_bsz)):
                # Build the Feed Dictionary, with inputs, outputs, dropout probability, and states.
                feed_dict = {rnn_langmod.X: x[start:end].reshape(bsz, steps),
                             rnn_langmod.Y: y[start:end].reshape(bsz, steps),
                             rnn_langmod.keep_prob: FLAGS.dropout_prob,
                             rnn_langmod.initial_state[0]: state[0],
                             rnn_langmod.initial_state[1]: state[1]}

                # Run the training operation with the Feed Dictionary, fetch loss and update state.
                curr_loss, _, state = sess.run([rnn_langmod.loss_val, rnn_langmod.train_op,
                                                rnn_langmod.final_state], feed_dict=feed_dict)

                # Update counters
                loss, iters = loss + curr_loss, iters + steps

                # Print Evaluation Statistics
                if start % FLAGS.eval_every == 0:
                    print 'Epoch {} Words {} to {} Perplexity: {}, took {} seconds!'.format(
                        epoch, start, end, np.exp(loss / iters), time.time() - start_time)
                    loss, iters = 0.0, 0

        # Evaluate Test Perplexity
        test_loss, test_iters, state = 0., 0, sess.run(rnn_langmod.initial_state)
        for s, e in zip(range(0, len(test_x - ex_bsz), ex_bsz), range(ex_bsz, len(test_x), ex_bsz)):
            # Build the Feed Dictionary, with inputs, outputs, dropout probability, and states.
            feed_dict = {rnn_langmod.X: test_x[s:e].reshape(bsz, steps),
                         rnn_langmod.Y: test_y[s:e].reshape(bsz, steps),
                         rnn_langmod.keep_prob: 1.0,
                         rnn_langmod.initial_state[0]: state[0],
                         rnn_langmod.initial_state[1]: state[1]}

            # Fetch the loss, and final state
            curr_loss, state = sess.run([rnn_langmod.loss_val, rnn_langmod.final_state],
                                        feed_dict=feed_dict)

            # Update counters
            test_loss, test_iters = test_loss + curr_loss, test_iters + steps

        # Print Final Output
        print 'Test Perplexity: {}'.format(np.exp(test_loss / test_iters))