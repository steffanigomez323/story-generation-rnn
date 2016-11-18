# """"
# Written by Steffani Gomez, smg1/@steffanigomez323
# This file will run the model with the given parameters.
# """
from __future__ import print_function
import tensorflow as tf
import argparse
from utils import TextReader
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type=str, default='smallshakespeare.txt',
                        help='the path to the input text')
    parser.add_argument('--lstm_size', type=int, default=128,
                        help='size of the LSTM hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--step_size', type=int, default=50,
                        help='the number of unrolled LSTM steps through backpropogation')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = TextReader(args.input_text, args.batch_size, args.step_size)
    args.vocab_size = data_loader.vocab_size
    print("The input file has been read and processed.")

    model = Model(lstm_size=args.lstm_size, batch_size=args.batch_size, step_size=args.step_size,
                  num_layers=args.num_layers, vocab_size=args.vocab_size, learning_rate=args.learning_rate)

    with tf.Session() as sess:
        print("Beginning to train the model...")
        model.train(sess, args.num_epochs, data_loader)
        print("The model has been trained.")


if __name__ == '__main__':
    main()