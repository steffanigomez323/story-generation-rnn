"""
Written by Steffani Gomez, smg1/@steffanigomez323
This file contains that are not directly related to the rnn-model, such as the text loading class.
"""

import sys

reload(sys)
sys.setdefaultencoding('utf8')
import codecs
import numpy as np
import re

class TextReader():

    def __init__(self, text, batchsz, numsteps, words, encoding="utf-8"):
        """
        This class handles reading the input text and creating the necessary vocabulary and its mapping for the
        different models.
        :param text: the text file
        :param batchsz: batch size
        :param numsteps: step size
        :param words: boolean denoting whether the vocabulary unit is words or characters
        :param encoding: the encoding in which the file should be read in
        """
        # type: (object, object) -> object
        self.input = text
        self.encoding = encoding
        self.batch_pointer = 0
        self.batch_size, self.step_size, self.step_batch_size = batchsz, numsteps, batchsz * numsteps
        self.processtext(words)
        #self.saveinputvocab(filename)

    def basic_tokenizer(self, sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
        """
        Very basic tokenizer: split the sentence into a list of tokens. Used in models where a vocabulary unit is a
        word.
        """
        words = []
        space_fragments = re.findall(r'\S+|\n', sentence)
        for space_separated_fragment in space_fragments:
            words.extend(re.split(word_split, space_separated_fragment))
        return [w for w in words if w]

    def processtext(self, words):
        """
        Creates the vocab, vocab size, mapping, and data from the input text and saves as instances of this class.
        :param words: flag denoting whether the vocabulary unit should be a word or a character
        :return: None
        """
        if words:
            # vocabulary is made up of words
            text = []
            with codecs.open(self.input, 'r', encoding=self.encoding) as f:
                for line in f:
                    words = self.basic_tokenizer(line)
                    for w in words:
                        text.append(w.encode('utf-8'))
            self.vocab = sorted(set(text))
            self.vocab_size = len(self.vocab)
            self.mapping = dict(zip(self.vocab, range(self.vocab_size)))
            self.data = np.array(list(map(self.mapping.get, text)))
            self.data_length = len(self.data)
            self.num_batches = int(self.data_length / (self.step_batch_size))

        else:
            # vocabulary is made up of characters
            with codecs.open(self.input, 'r', encoding=self.encoding) as f:
                text = f.read().encode('utf-8')
            self.vocab = list(''.join(sorted(set(text))))
            self.vocab_size = len(self.vocab)
            self.mapping = dict(zip(self.vocab, range(self.vocab_size)))
            self.data = np.array(list(map(self.mapping.get, text)))
            self.data_length = len(self.data)
            self.num_batches =int(self.data_length / (self.step_batch_size))

    def nextbatch(self):
        """
        This method returns the next batch of words and their targets according to the batch pointer
        :return: the next batch to avoid storing storing everything in the tensorflow graph and running out of memory
        """
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
