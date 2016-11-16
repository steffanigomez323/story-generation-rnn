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
langmod_rnn2.py

Non-object oriented version of Tensorflow code for the LSTM Language Model.
"""
from collections import defaultdict
import numpy as np
import re
import tensorflow as tf
import time

BOOK_PATH = "dracula.txt"
STOP = "*STOP*"
UNK = "*UNK*"
UNK_ID = 0
STOP_ID = 1

MAX_VOCAB_SIZE = 8000
BSZ = 50
NUM_STEPS = 20
EMBEDDING_SIZE = 50
HIDDEN_SIZE = 256
DROPOUT_PROB = 0.5
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
EVAL_EVERY = 10000


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
            raw_tokens.extend(words)

    # Add STOP Symbols after Periods
    stop_list = []
    for i in raw_tokens:
        if i == '.':
            stop_list.extend([i, STOP])
        else:
            stop_list.append(i)

    # Create the vocabulary
    vocab_list = [UNK, STOP] + sorted(vocabulary, key=vocabulary.get, reverse=True)
    vocabulary = {vocab_list[i]: i for i in range(MAX_VOCAB_SIZE) if i < len(vocab_list)}

    # Use the vocabulary to vectorize the data, return x, y, and vocabulary
    data = map(lambda tok: vocabulary.get(tok, UNK_ID), stop_list)
    train_len = int(len(data) * 0.9)
    return np.array(data[:train_len - 1], dtype=int), np.array(data[1:train_len], dtype=int), \
        np.array(data[train_len:-1], dtype=int), np.array(data[train_len + 1:], dtype=int), \
        vocabulary, vocab_list

def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# Preprocess and vectorize the data
x, y, test_x, test_y, vocab, vocab_list = read()

# Setup Placeholders
X = tf.placeholder(tf.int32, shape=[None, NUM_STEPS])
Y = tf.placeholder(tf.int32, shape=[None, NUM_STEPS])
keep_prob = tf.placeholder(tf.float32)

# Embedding Matrix
E = weight_variable([MAX_VOCAB_SIZE, EMBEDDING_SIZE], 'Embedding')

# Basic LSTM Cell
cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
initial_state = cell.zero_state(BSZ, tf.float32)

# Softmax Output
softmax_w = weight_variable([HIDDEN_SIZE, MAX_VOCAB_SIZE], 'Softmax_Weight')
softmax_b = weight_variable([MAX_VOCAB_SIZE], 'Softmax_Bias')

# Feed input through the Embedding Layer, Dropout.
emb = tf.nn.embedding_lookup(E, X)                                   # Shape [bsz, steps, hidden]
drop_emb = tf.nn.dropout(emb, keep_prob)

# Feed input through dynamic_rnn
out, f_state = tf.nn.dynamic_rnn(cell, drop_emb,                     # Shape [bsz, steps, hidden]
                                 initial_state=initial_state)

# Reshape the outputs into a single 2D Tensor
outputs = tf.reshape(out, [-1, HIDDEN_SIZE])                         # Shape [bsz * steps, hidden]

# Feed through final layer, compute logits
logits = tf.matmul(outputs, softmax_w) + softmax_b                   # Shape [bsz * steps, vocab]

# Compute Loss
seq_loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(Y, [-1])],
                                                  [tf.ones([BSZ * NUM_STEPS])])
loss_val = tf.reduce_sum(seq_loss) / BSZ

# Training Operation
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_val)

# Launch Tensorflow Session
print 'Launching Tensorflow Session'
sess = tf.Session()

# Initialize all Variables
sess.run(tf.initialize_all_variables())

# Start Training
ex_bsz, bsz, steps = BSZ * NUM_STEPS, BSZ, NUM_STEPS
for epoch in range(NUM_EPOCHS):
    state, loss, iters, start_time = sess.run(initial_state), 0., 0, time.time()
    for start, end in zip(range(0, len(x) - ex_bsz, ex_bsz), range(ex_bsz, len(x), ex_bsz)):
        # Build the Feed Dictionary, with inputs, outputs, dropout probability, and states.
        feed_dict = {X: x[start:end].reshape(bsz, steps),
                     Y: y[start:end].reshape(bsz, steps),
                     keep_prob: DROPOUT_PROB,
                     initial_state: state}

        # Run the training operation with the Feed Dictionary, fetch loss and update state.
        curr_loss, _, state = sess.run([loss_val, train_op, f_state], feed_dict=feed_dict)

        # Update counters
        loss, iters = loss + curr_loss, iters + steps

        # Print Evaluation Statistics
        if start % EVAL_EVERY == 0:
            print 'Epoch {} Words {} to {} Perplexity: {}, took {} seconds!'.format(
                epoch, start, end, np.exp(loss / iters), time.time() - start_time)
            loss, iters = 0.0, 0

# Evaluate Test Perplexity
test_loss, test_iters, state = 0., 0, sess.run(initial_state)
for s, e in zip(range(0, len(test_x - ex_bsz), ex_bsz), range(ex_bsz, len(test_x), ex_bsz)):
    # Build the Feed Dictionary, with inputs, outputs, dropout probability, and states.
    feed_dict = {X: test_x[s:e].reshape(bsz, steps),
                 Y: test_y[s:e].reshape(bsz, steps),
                 keep_prob: 1.0,
                 initial_state: state}

    # Fetch the loss, and final state
    curr_loss, state = sess.run([loss_val, f_state], feed_dict=feed_dict)

    # Update counters
    test_loss, test_iters = test_loss + curr_loss, test_iters + steps

# Print Final Output
print 'Test Perplexity: {}'.format(np.exp(test_loss / test_iters))