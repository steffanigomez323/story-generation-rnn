# """"
# Written by Steffani Gomez, smg1/@steffanigomez323
# This file will run the model with the given parameters.
# """
from __future__ import print_function
import tensorflow as tf
import argparse
from utils import TextReader
from model import Model
import cPickle as pickle
import codecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='xfilesseason1',
                        help='the name of the model')
    parser.add_argument('--input_text', type=str, default='data/xfilesseason1.txt',
                        help='the path to the input text')
    parser.add_argument('--words', type=bool, default=True,
                        help='boolean variable that determines whether genereate text by word or by character')
    parser.add_argument('--output_dir', type=str, default='output/',
                        help='the directory in which to place the output file.')
    parser.add_argument('--lstm_size', type=int, default=128,
                        help='size of the LSTM hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='minibatch size')
    parser.add_argument('--step_size', type=int, default=100,
                        help='the number of unrolled LSTM steps through backpropogation')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--save_vocab_file', type=str, default='save/vocab/',
                        help='the location in which to save the vocab pkl file')
    parser.add_argument('--save_model_file', type=str, default='save/models/',
                        help='the location in which to save the model pkl file')
    args = parser.parse_args()
    args.save_vocab_file = args.save_vocab_file + args.label + '.pkl'
    args.save_model_file = args.save_model_file + args.label + '_' + str(args.batch_size) + '_' + str(args.step_size) + "_" +\
                           str(args.num_epochs) + '_' + str(args.num_layers) + '_' + str(args.lstm_size) + '.ckpt'
    args.output_file = args.output_dir + args.label + '_' + str(args.batch_size) + '_' + str(args.step_size) + "_" +\
                           str(args.num_epochs) + '_' + str(args.num_layers) + '_' + str(args.lstm_size) + '.txt'

    #textloader = train(args)
    textloader = TextReader(args.input_text, args.batch_size, args.step_size, args.words, args.save_vocab_file)
    samplefrommodel(args, textloader)


def train(args):
    #with codecs.open(args.save_vocab_file) as vocab:
    #    textloader = pickle.load(vocab)
    textloader = TextReader(args.input_text, args.batch_size, args.step_size, args.words, args.save_vocab_file)
    print("Vocab has been loaded.")
    #with codecs.open(args.save_vocab_file, 'w+', encoding="UTF-8", errors="ignore") as f:
    #    pickle.dump(textloader, f)
    args.vocab_size = textloader.vocab_size
    model = Model(lstm_size=args.lstm_size, batch_size=args.batch_size, step_size=args.step_size,
                  num_layers=args.num_layers, vocab_size=args.vocab_size, learning_rate=args.learning_rate)
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        print("Beginning to train the model...")
        model.train(sess, args.num_epochs, textloader)
        print("The model has been trained.")
        save_path = saver.save(sess, args.save_model_file)
        print("Model saved in file: %s" % save_path)
    tf.reset_default_graph()
    return textloader

def samplefrommodel(args, textloader):
    #with codecs.open(args.save_vocab_file) as vocab:
    #    textloader = pickle.load(vocab)
    print("Vocab has been loaded.")
    args.vocab_size = textloader.vocab_size
    #model = pickle.open(args.save_model_file)
    #print("Model has been loaded.")
    args.batch_size = 1
    args.step_size = 1
    model = Model(lstm_size=args.lstm_size, batch_size=args.batch_size, step_size=args.step_size,
                  num_layers=args.num_layers, vocab_size=args.vocab_size, learning_rate=args.learning_rate)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, args.save_model_file)
        print("Model has been loaded.")
        sampled_text = model.sample(sess=sess, vocab=textloader.vocab, vocabmapping=textloader.mapping)
        with codecs.open(args.output_file, 'w+', encoding='UTF-8') as f:
            f.write(sampled_text)
        print(sampled_text)
        print("Model has been sampled from.")



if __name__ == '__main__':
    main()
