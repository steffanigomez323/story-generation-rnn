"""
Written by Steffani Gomez, smg1/@steffanigomez323
This file contains any classes that are not directly related to the rnn-model, such as the text loading class.
"""
import sys

reload(sys)
sys.setdefaultencoding('utf8')
import codecs
import numpy as np
import cPickle as pickle

class TextReader():

    def __init__(self, text, batchsz, numsteps, filename, encoding="utf-8"):
        # type: (object, object) -> object
        self.input = text
        self.encoding = encoding
        self.batch_poointer = 0
        self.batch_size, self.step_size, self.step_batch_size = batchsz, numsteps, batchsz * numsteps
        self.processtext()
        #self.saveinputvocab(filename)

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

    def nextbatch(self):
        # returns the next batch to avoid storing everything in memory and running out of memory
        x = np.array(self.data[:self.data_length - 1], dtype=int)
        y = np.array(self.data[1:self.data_length], dtype=int)
        xdata = x[self.batch_pointer:self.batch_pointer + self.step_batch_size].reshape(self.batch_size, self.step_size)
        ydata = y[self.batch_pointer:self.batch_pointer + self.step_batch_size].reshape(self.batch_size, self.step_size)
        self.batch_pointer += self.step_batch_size
        return xdata, ydata

    def resetpointer(self):
        """
        resets the batch pointer that defines what batch to return as the next batch to 0
        :return: nothing
        """
        self.batch_pointer = 0


    def saveinputvocab(self, filename):
        """
        Saves this class as in a pickle file so that it can be restored later if necessary.
        :param filename: the name of the file in which to place the pickle object
        :return: nothing
        """
        with codecs.open(filename, 'w+', encoding=self.encoding, errors='replace') as f:
            pickle.dump(self, f)