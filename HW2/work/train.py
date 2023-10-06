import pickle
import argparse
from neural_model import NeuralModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net training arguments.')

    parser.add_argument('-u', type=int, help='number of hidden units')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-f', type=int, help='max sequence length')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-E', type=str, help='word embedding file')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('-o', type=str, help='model file to be written')

    args = parser.parse_args()
    
    # Get paramaters

    model = NeuralModel(
        embeddingfile=args.E,
        max_seq_length=args.f,
        hidden_units=args.u,
        minibatch_size=args.b,
        learning_rate=args.l,
        epochs=args.e
    )
    
    model.train(args.i)
    
    model.save_model(args.o)
