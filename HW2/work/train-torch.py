import pickle
import argparse
from neural_model import NeuralModel_Torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net training arguments.')

    parser.add_argument('-u', type=int, help='number of hidden units')
    parser.add_argument('-ul',
                        type=lambda s: [int(item) for item in s.split(',')], default=[],
                        help='number of hidden units in each layer. Example: "10,5"'
    ) # Convert comma separated string to list
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-f', type=int, help='max sequence length')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-E', type=str, help='word embedding file')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('-o', type=str, help='model file to be written')
    parser.add_argument('-adam', type=str, default=False, help='Enable Adam optimizer')
    parser.add_argument('-average_emb_sentence', type=bool, default=False, help="Compute the average of the embeddings in the sentence instead of concatenation")

    args = parser.parse_args()
    
    # Get paramaters

    model = NeuralModel_Torch(
        embeddingfile=args.E,
        max_seq_length=args.f,
        hidden_units=args.u,
        minibatch_size=args.b,
        learning_rate=args.l,
        epochs=args.e,
        adam=args.adam,
        hidden_units_other_layers=args.ul,
        average_emb_sentence=args.average_emb_sentence
    )
     
    model.train(args.i)
    
    model.save_model(args.o)
