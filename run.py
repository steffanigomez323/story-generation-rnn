# """"
# Written by Steffani Gomez, smg1/@steffanigomez323
# This file will run the model with the given parameters.
# """
from __future__ import print_function
import tensorflow as tf
import argparse
from textloader import TextReader
from models import RNNModel
from models import LDAModel
import cPickle as pickle
import codecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', type=str,
                        help='the name of the model')
    parser.add_argument('model', type=str,
                        help='which model to run, rnn or lda')
    parser.add_argument('-train', type=str, default='true',
                        help='boolean string that determines whether the model should be trained or loaded'
                             ' from disk')
    parser.add_argument('-num_topics', type=int, default=35,
                        help='how many topics to find in the input text')
    parser.add_argument('-num_words', type=int, default=2000,
                        help='number of words to generate from model')
    parser.add_argument('-doclen', type=int, default=50,
                        help='the length of the a document in the lda model')
    parser.add_argument('input_text', type=str,
                        help='the path to the input text')
    parser.add_argument('-words', type=str, default='true',
                        help='determines whether generate text by word or by character')
    parser.add_argument('output_dir', type=str,
                        help='the directory in which to place the output file.')
    parser.add_argument('-lstm_size', type=int, default=128,
                        help='size of the LSTM hidden state')
    parser.add_argument('-num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('-batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('-step_size', type=int, default=50,
                        help='the number of unrolled LSTM steps through backpropogation')
    parser.add_argument('-num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('-learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('-save_model_dir', type=str, default='save/models/',
                        help='the directory in which to save/load the model file')
    args = parser.parse_args()

    if args.train.lower() == 'false':
        args.train = False
    elif args.train.lower() == 'true':
        args.train = True

    if args.words.lower() == 'false':
        args.words = False
    elif args.words.lower() == 'true':
        args.words = True

    if args.model == 'lda':

        args.save_model_file = args.save_model_dir + args.model + '_' + args.label + '_' + str(args.doclen) + '_' +  \
                               str(args.num_epochs) + '_' + str(args.num_topics) + '.pkl'
        args.output_file = args.output_dir + args.model + '_' + args.label + '_' + str(args.doclen) + '_' +  \
                               str(args.num_epochs) + '_' + str(args.num_topics) + '.txt'

        ldamodel = LDAModel(args.input_text, args.num_topics, args.num_epochs, args.doclen)

        try:
            with codecs.open(args.save_model_dir, 'wb', encoding="UTF-8", errors="ignore") as f:
               pickle.dump(ldamodel, f)
            print("Model saved in file: %s" % args.save_model_file)
        except UnicodeDecodeError:
            print("Model was unable to be saved due to a utf-8 encoding error, please make sure your input text is"
                  "UTF-8 encoded.")

        results = ldamodel.generatetext(args.num_words)
        results = ' '.join(results)
        print(results)
        with codecs.open(args.output_file, 'w+', encoding='UTF-8') as f:
            f.write(results)

    else:

        args.save_model_file = args.save_model_dir + args.label + '_' + str(
            args.batch_size) + '_' + str(
            args.step_size) + "_" + \
                               str(args.num_epochs) + '_' + str(args.num_layers) + '_' + str(
            args.lstm_size) + '_' + str(
            args.words) + '.ckpt'
        args.output_file = args.output_dir + args.label + '_' + str(args.batch_size) + '_' + str(
            args.step_size) + "_" + \
                           str(args.num_epochs) + '_' + str(args.num_layers) + '_' + str(args.lstm_size) + '_' + str(
            args.words) + '.txt'

        textloader = TextReader(args.input_text, args.batch_size, args.step_size, args.words)
        print("Vocab has been loaded.")
        if args.train:
            train(args, textloader)
        samplefrommodel(args, textloader)


def train(args, textloader):
    args.vocab_size = textloader.vocab_size
    model = RNNModel(lstm_size=args.lstm_size, batch_size=args.batch_size, step_size=args.step_size,
                  num_layers=args.num_layers, vocab_size=args.vocab_size, learning_rate=args.learning_rate)
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        print("Beginning to train the model...")
        model.train(sess, args.num_epochs, textloader)
        print("The model has been trained.")
        save_path = saver.save(sess, args.save_model_file)
        print("Model saved in file: %s" % save_path)
    tf.reset_default_graph()

def samplefrommodel(args, textloader):
    args.vocab_size = textloader.vocab_size
    args.batch_size = 1
    args.step_size = 1
    model = RNNModel(lstm_size=args.lstm_size, batch_size=args.batch_size, step_size=args.step_size,
                  num_layers=args.num_layers, vocab_size=args.vocab_size, learning_rate=args.learning_rate)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, args.save_model_file)
        print("Model has been loaded.")
        if args.words:
            sampled_text = model.sample(sess=sess, vocab=textloader.vocab, words=args.words, vocabmapping=textloader.mapping)
        else:
            sampled_text = model.sample(sess=sess, vocab=textloader.vocab, words=args.words, prime=' ', vocabmapping=textloader.mapping)
        with codecs.open(args.output_file, 'w+', encoding='UTF-8') as f:
            f.write(sampled_text)
        print(sampled_text)
        print("Model has been sampled from.")



if __name__ == '__main__':
    main()
