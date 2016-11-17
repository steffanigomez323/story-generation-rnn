""""
Written by Steffani Gomez, smg1/@steffanigomez323
This file will run the model with the given parameters.
"""
import argparse
from utils import TextReader
from model import Model
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, default='smallshakespeare.txt',
                        help='the input text')
    parser.add_argument('--lstm_size', type=int, default=256,
                        help='the size of the LSTM layer')
    parser.add_argument('--step_size', type=int, default=20,
                        help='number of unrolled steps')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='the size of a training batch')
    parser.add_argument('--learning_rate', type=int, default=1e-4,
                        help='the rate at which the model learns')
    parser.add_argument('--dropout_rate', type=int, default=0.5,
                        help='the keep probability for dropout')
    parser.add_argument('--embedding_size', type=int, default=50,
                        help='the embedding size')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='the number of epochs to train over')
    parser.add_argument('--eval_every', type=int, default=10000,
                        help='when to evaluate how the model is doing after this many training steps')
    args = parser.parse_args()
    reader = TextReader(args.input_txt)
    print("Data has been read in and processed.")
    model = Model(batchsz=args.batch_size, dropout=args.dropout_rate, lstmsz=args.lstm_size,
                  embeddingsz=args.embedding_size, learningrate=args.learning_rate, numepochs=args.num_epochs,
                  stepsz=args.step_size, vocabsz=reader.data_length)
    with tf.Session() as sess:
        print("Starting to train the LSTM model.")
        model.train(args.eval_every, sess, reader)
        print("Training has been completed.")

if __name__ == '__main__':
    main()