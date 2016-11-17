"""
Written by Steffani Gomez/@steffanigomez323
This file contains the rnn model that will be trained and then sampled from to generate text
"""
import numpy as np
import tensorflow as tf
import time

class Model():

    def __init__(self, batchsz, lstmsz, numepochs, stepsz, learningrate, vocabsz, embeddingsz, dropout):
        self.batch_size, self.hidden_size = batchsz, lstmsz
        self.num_epochs, self.step_size = numepochs, stepsz
        self.learning_rate, self.vocab_size, self.embedding_size = learningrate, vocabsz, embeddingsz
        self.dropout_rate = dropout

        # setting up placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.step_size])
        self.Y = tf.placeholder(tf.int32, shape=[None, self.step_size])
        self.keep_prob = tf.placeholder(tf.float32)

        # Embedding Matrix
        self.E = self.weight_variable([self.vocab_size, self.embedding_size], 'Embedding')

        # Basic LSTM Cell
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        # Softmax Output
        self.softmax_w = self.weight_variable([self.hidden_size, self.vocab_size], 'Softmax_Weight')
        self.softmax_b = self.weight_variable([self.vocab_size], 'Softmax_Bias')

        # Feed input through the Embedding Layer, Dropout.
        emb = tf.nn.embedding_lookup(self.E, self.X)  # Shape [bsz, steps, hidden]
        drop_emb = tf.nn.dropout(emb, self.keep_prob)

        # Feed input through dynamic_rnn
        out, self.final_state = tf.nn.dynamic_rnn(self.cell, drop_emb,  # Shape [bsz, steps, hidden]
                                         initial_state=self.initial_state)

        # Reshape the outputs into a single 2D Tensor
        outputs = tf.reshape(out, [-1, self.hidden_size])  # Shape [batch_size * step_size, hidden_size]

        # Feed through final layer, compute logits
        self.logits = tf.matmul(outputs, self.softmax_w) + self.softmax_b  # Shape [bsz * steps, vocab]

        # Build the Loss Computation
        self.loss_val = self.loss()

        # Build the Training Operation
        self.train_op = self.train_step()

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def loss(self):
        """
        Build the sequence cross-entropy loss by example computation.

        :return Scalar representing sequence loss.
        """
        seq_loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits],
                                                          [tf.reshape(self.Y, [-1])],
                                                          [tf.ones([self.batch_size * self.step_size])])
        loss = tf.reduce_sum(seq_loss) / self.batch_size
        return loss

    def train_step(self):
        """
        Build the training operation, using the sequence loss by example and an Adam Optimizer.

        :return Training Operation
        """
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss_val)

    def train(self, eval_every, sess, textreader):
        # Preprocess and vectorize the data
        #x, y, test_x, test_y, vocab, vocab_list = read()
        x, y = textreader.firstbatch()
        # Launch Tensorflow Session
        print 'Launching Tensorflow Session'
        #with tf.Session() as sess:
        # Instantiate Model
        #rnn_langmod = RNNLangmod(len(vocab), FLAGS.embedding_size, FLAGS.num_steps,
        #                        FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

        # Initialize all Variables
        sess.run(tf.initialize_all_variables())

        # Start Training
        ex_bsz, bsz, steps = self.batch_size * self.step_size, self.batch_size, self.step_size
        for epoch in range(self.num_epochs):
            state, loss, iters, start_time = sess.run(self.initial_state), 0., 0, time.time()
            for start, end in zip(range(0, len(x) - ex_bsz, ex_bsz), range(ex_bsz, len(x), ex_bsz)):
                # Build the Feed Dictionary, with inputs, outputs, dropout probability, and states.
                feed_dict = {self.X: x[start:end].reshape(bsz, steps),
                             self.Y: y[start:end].reshape(bsz, steps),
                             self.keep_prob: self.dropout_rate,
                             self.initial_state[0]: state[0],
                             self.initial_state[1]: state[1]}

                # Run the training operation with the Feed Dictionary, fetch loss and update state.
                curr_loss, _, state = sess.run([self.loss_val, self.train_op,
                                                self.final_state], feed_dict=feed_dict)

                # Update counters
                loss, iters = loss + curr_loss, iters + steps

                # Print Evaluation Statistics
                if start % eval_every == 0:
                    print 'Epoch {} Words {} to {} Perplexity: {}, took {} seconds!'.format(
                        epoch, start, end, np.exp(loss / iters), time.time() - start_time)
                    loss, iters = 0.0, 0

    def test(self, sess, ex_bsz, test_x, test_y):
        # Evaluate Test Perplexity
        test_loss, test_iters, state = 0., 0, sess.run(self.initial_state)
        for s, e in zip(range(0, len(test_x - ex_bsz), ex_bsz), range(ex_bsz, len(test_x), ex_bsz)):
            # Build the Feed Dictionary, with inputs, outputs, dropout probability, and states.
            feed_dict = {self.X: test_x[s:e].reshape(self.batch_size, self.step_size),
                         self.Y: test_y[s:e].reshape(self.batch_size, self.step_size),
                         self.keep_prob: 1.0,
                         self.initial_state[0]: state[0],
                         self.initial_state[1]: state[1]}

            # Fetch the loss, and final state
            curr_loss, state = sess.run([self.loss_val, self.final_state],
                                        feed_dict=feed_dict)

            # Update counters
            test_loss, test_iters = test_loss + curr_loss, test_iters + self.step_size

        # Print Final Output
        print 'Test Perplexity: {}'.format(np.exp(test_loss / test_iters))

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret