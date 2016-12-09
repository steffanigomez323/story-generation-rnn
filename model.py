# """
# Written by Steffani Gomez/@steffanigomez323
# This file contains the rnn model that will be trained and then sampled from to generate text
# """
import tensorflow as tf
import numpy as np
import time
import codecs
import gensim
from gensim import corpora
import string
import re

class Model():
    def __init__(self, lstm_size, num_layers, batch_size, step_size, vocab_size, learning_rate):
        cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        #else:
        #    raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(lstm_size, state_is_tuple=True)

        # this command creates a stacked LSTM model with the specified number of layers
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        # the input data, also known as the x vector
        self.X = tf.placeholder(tf.int32, [batch_size, step_size])

        # the output data, also known as the y vector
        self.Y = tf.placeholder(tf.int32, [batch_size, step_size])

        # the initial state of the model
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # creates a variable graph that can be reused, and initializes the variables
        with tf.variable_scope('rnnlm'):

            # the softmax weights
            softmax_w = tf.get_variable("softmax_w", [lstm_size, vocab_size])

            # the softmax biases
            softmax_b = tf.get_variable("softmax_b", [vocab_size])

            # this creates a graph that uses the computer's CPU
            with tf.device("/cpu:0"):

                # the embedding matrix
                embedding = tf.get_variable("embedding", [vocab_size, lstm_size])

                # feeding the input data through the embedding layer
                inputs = tf.split(1, step_size, tf.nn.embedding_lookup(embedding, self.X))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, lstm_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.loss_val = self.loss(batch_size, step_size, vocab_size)
        self.final_state = last_state
        self.train_op = self.train_step(learning_rate)

    def loss(self, batch_size, step_size, vocab_size):
        l = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.Y, [-1])], [tf.ones([batch_size * step_size])],
                                         vocab_size)
        return tf.reduce_sum(l) / batch_size / step_size

    def train_step(self, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.loss_val)

    def train(self, sess, num_epochs, data_loader):
        tf.initialize_all_variables().run()
        for e in range(num_epochs):
            data_loader.resetpointer()
            state = sess.run(self.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.nextbatch()
                feed = {self.X: x, self.Y: y}
                for i, (c, h) in enumerate(self.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                train_loss, state, _ = sess.run([self.loss_val, self.final_state, self.train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.num_batches + b,
                              num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))

    #def savemodel(self, filename):
    #    with codecs.open(filename, 'w+') as f:
    #        pickle.dump(self, f)

    def sample(self, sess, vocab, vocabmapping, num=5000, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for v in prime.split()[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocabmapping[v]
            feed = {self.X: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        v = prime.split()[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocabmapping[v]
            feed = {self.X: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if v == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = vocab[sample]
            ret += " " + pred
            v = pred
        return ret.encode('utf-8')


# state = sess.run(self.cell.zero_state(1, tf.float32))
# if not len(prime) or prime == " ":
#     prime = random.choice(list(vocab.keys()))
# print (prime)
# for word in prime.split()[:-1]:
#     print (word)
#     x = np.zeros((1, 1))
#     x[0, 0] = vocab.get(word, 0)
#     feed = {self.input_data: x, self.initial_state: state}
#     [state] = sess.run([self.final_state], feed)
#
#
# def weighted_pick(weights):
#     t = np.cumsum(weights)
#     s = np.sum(weights)
#     return (int(np.searchsorted(t, np.random.rand(1) * s)))
#
#
# ret = prime
# word = prime.split()[-1]
# for n in range(num):
#     x = np.zeros((1, 1))
#     x[0, 0] = vocab.get(word, 0)
#     feed = {self.input_data: x, self.initial_state: state}
#     [probs, state] = sess.run([self.probs, self.final_state], feed)
#     p = probs[0]
#
#     if sampling_type == 0:
#         sample = np.argmax(p)
#     elif sampling_type == 2:
#         if word == '\n':
#             sample = weighted_pick(p)
#         else:
#             sample = np.argmax(p)
#     else:  # sampling_type == 1 default:
#         sample = weighted_pick(p)
#
#     pred = words[sample]
#     ret += ' ' + pred
#     word = pred
# return ret


class LDAModel():

    def __init__(self, file, numtopics, epochs, doclen):
        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        docs = []
        linecount = 0
        with codecs.open(file, 'r', encoding='utf-8') as f:
            for line in f:
                words = self.basic_tokenizer(line)
                for w in words:
                    docs.append(w.encode('utf-8'))
                linecount += 1

        docs_complete = []
        #doclen = np.random.poisson(lam=int(10), size=1)
        #doclen = 10
        for i in range(0, len(docs), doclen):
            docs_complete.append(' '.join(docs[i:i + doclen]))


        docs_complete_clean = [doc.encode('utf-8').split(" ") for doc in docs_complete]

        dictionary = corpora.Dictionary(docs_complete_clean)
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_complete_clean]

        # Running and Training LDA model on the document term matrix.
        self.model = Lda(doc_term_matrix, num_topics=numtopics, id2word=dictionary, passes=epochs)

        wholetext = []

        for doc in docs_complete:
            for w in self.basic_tokenizer(doc):
                wholetext.append(w.encode('utf-8'))

        self.topics = self.model.get_document_topics(dictionary.doc2bow(wholetext), minimum_probability=0)
        self.mapping = dictionary


    def basic_tokenizer(self, sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
        """
        Very basic tokenizer: split the sentence into a list of tokens.
        """
        words = []
        space_fragments = re.findall(r'\S+|\n', sentence)
        for space_separated_fragment in space_fragments:
            # for space_separated_fragment in sentence.split():
            words.extend(re.split(word_split, space_separated_fragment))
        return [w for w in words if w]

    def generatetext(self, numwords=50):
        res = []
        for i in range(numwords):
            topic = np.random.choice([int(i[0]) for i in self.topics], p=[float(i[1]) for i in self.topics])
            worddist = self.model.get_topic_terms(topic, len(self.mapping.keys()))
            word = np.random.choice([int(i[0]) for i in worddist], p=[float(i[1]) for i in worddist])
            res.append(self.mapping.get(word).encode('utf-8'))
        #print(' '.join(res))
        return res