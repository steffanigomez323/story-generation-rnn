"""
Written by Steffani Gomez, smg1/@steffanigomez323
This file contains any classes that are not directly related to the rnn-model, such as the text loading class.
"""
import codecs
import numpy as np

class TextReader():

    def __init__(self, text, encoding="utf-8"):
        # type: (object, object) -> object
        self.input = text
        self.encoding = encoding
        self.processtext()

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

    def firstbatch(self):
        return np.array(self.data[:self.data_length - 1], dtype=int), np.array(self.data[1:self.data_length])
