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
import re

class RNNModel():
    def __init__(self, lstm_size, num_layers, batch_size, step_size, vocab_size, learning_rate):
        """
        Initiating this class involves setting up the tensorflow graph and its operations
        :param lstm_size: the size of the lstm
        :param num_layers: the number of layers
        :param batch_size: the batch size
        :param step_size: the step size
        :param vocab_size: the size of the vocabulary
        :param learning_rate: the learning rate for the training operation
        """
        cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)

        # creates an RNN with multiple layers
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        # input data
        self.X = tf.placeholder(tf.int32, [batch_size, step_size])

        # target data
        self.Y = tf.placeholder(tf.int32, [batch_size, step_size])

        # the initial state of the model
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # creates a variable graph that can be reused, and initializes the variables
        with tf.variable_scope('rnnlstm'):

            # the softmax weights
            w = tf.get_variable("w", [lstm_size, vocab_size])

            # the softmax biases
            b = tf.get_variable("b", [vocab_size])

            # this creates a graph that uses the computer's CPU
            with tf.device("/cpu:0"):

                # the embedding matrix
                embedding = tf.get_variable("embedding", [vocab_size, lstm_size])

                # feeding the input data through the embedding layer
                inputs = tf.split(1, step_size, tf.nn.embedding_lookup(embedding, self.X))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlstm')
        output = tf.reshape(tf.concat(1, outputs), [-1, lstm_size])
        self.logits = tf.matmul(output, w) + b
        self.probability = tf.nn.softmax(self.logits)
        self.loss_val = self.loss(batch_size, step_size, vocab_size)
        self.final_state = last_state
        self.train_op = self.train_step(learning_rate)

    def loss(self, batch_size, step_size, vocab_size):
        """
        The loss function on which to minimize
        :param batch_size: the batch size
        :param step_size: the step size
        :param vocab_size: the size of the vocabulary
        :return: None
        """
        l = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.Y, [-1])], [tf.ones([batch_size * step_size])],
                                         vocab_size)
        return tf.reduce_sum(l) / batch_size

    def train_step(self, learning_rate):
        """
        This is the training operation in the tensorflow graph
        :param learning_rate: the learning rate for the training operation
        :return: None
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.loss_val)

    def train(self, sess, num_epochs, data_loader):
        """
        This method trains the model with the set number of epochs
        :param sess: the tensorflow session
        :param num_epochs: the number of epochs to train for
        :param data_loader: the textloader class that contains the input data
        :return: None
        """
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
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(e * data_loader.num_batches + b,
                              num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))

    def sample(self, sess, vocab, vocabmapping, words, num=5000, prime='The '):
        """
        This method runs the model with a beginning input and samples from the probabilities of the model to
        generate text
        :param sess: the tensorflow session
        :param vocab: the vocabulary
        :param vocabmapping: the vocabulary to integer mapping
        :param words: boolean denoting whether a vocabulary unit is a word or a character
        :param num: the number of vocabulary units to generate
        :param prime: the beginning input to start the model off with
        :return: a string that was generated from the model
        """
        state = sess.run(self.cell.zero_state(1, tf.float32))
        if words:
            primeinput = prime.split()[:-1]
        else:
            primeinput = prime[:-1]

        for v in primeinput:
            x = np.zeros((1, 1))
            x[0, 0] = vocabmapping[v]
            feed = {self.X: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        res = prime
        v = primeinput
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocabmapping[v]
            feed = {self.X: x, self.initial_state:state}
            [probs, state] = sess.run([self.probability, self.final_state], feed)
            p = probs[0]
            sample = weighted_pick(p)
            res += " " + vocab[sample]
            v = vocab[sample]
        return res.encode('utf-8')

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
            words.extend(re.split(word_split, space_separated_fragment))
        return [w for w in words if w]

    def generatetext(self, numwords=50):
        """
        Generates text according to the topic distribution of the text and then the word probabilities of the chosen topic
        :param numwords: the number of words to generate
        :return: an array of words that were generated from the model
        """
        res = []
        for i in range(numwords):
            topic = np.random.choice([int(i[0]) for i in self.topics], p=[float(i[1]) for i in self.topics])
            worddist = self.model.get_topic_terms(topic, len(self.mapping.keys()))
            word = np.random.choice([int(i[0]) for i in worddist], p=[float(i[1]) for i in worddist])
            res.append(self.mapping.get(word).encode('utf-8'))
        return res