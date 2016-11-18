"""
Written by Steffani Gomez, smg1/@steffanigomez323
This file contains any classes that are not directly related to the rnn-model, such as the text loading class.
"""
import codecs
import numpy as np

class TextReader():

    def __init__(self, text, batchsz, numsteps, encoding="utf-8"):
        # type: (object, object) -> object
        self.input = text
        self.encoding = encoding
        self.batch_poointer = 0
        self.batch_size, self.step_size, self.step_batch_size = batchsz, numsteps, batchsz * numsteps
        self.processtext()
        #self.create_batches()

    def processtext(self):
        """
        Processes the text, saves the vocab, vocab size, and mapping as instance variables of this class
        :return: a np array of the input text with characters replaced by their corresponding integers
        """
        with codecs.open(self.input, 'r', encoding=self.encoding) as f:
            text = f.read()
        self.chars = list(''.join(sorted(set(text))))
        self.vocab_size = len(self.chars)
        self.mapping = dict(zip(self.chars, range(self.vocab_size)))
        self.data = np.array(list(map(self.mapping.get, text)))
        self.data_length = len(self.data)
        self.num_batches =int(self.data_length / (self.step_batch_size))

    # def create_batches(self):
    #     # When the data (tensor) is too small, let's give them a better error message
    #     if self.num_batches==0:
    #         assert False, "Not enough data. Make seq_length and batch_size small."
    #
    #     self.data = self.data[:self.num_batches * self.step_batch_size]
    #     xdata = self.data
    #     ydata = np.copy(self.data)
    #     ydata[:-1] = xdata[1:]
    #     ydata[-1] = xdata[0]
    #     self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
    #     self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def nextbatch(self):
        # returns the next batch to avoid storing everything in memory and running out of memory
        x = np.array(self.data[:self.data_length - 1], dtype=int)
        y = np.array(self.data[1:self.data_length], dtype=int)
        xdata = x[self.batch_pointer:self.batch_pointer + self.step_batch_size].reshape(self.batch_size, self.step_size)
        ydata = y[self.batch_pointer:self.batch_pointer + self.step_batch_size].reshape(self.batch_size, self.step_size)
        self.batch_pointer += self.step_batch_size
        return xdata, ydata
        # x, y = self.x_batches[self.batch_pointer], self.y_batches[self.batch_pointer]
        # self.batch_pointer += 1
        # return x, y

    def resetpointer(self):
        self.batch_pointer = 0


