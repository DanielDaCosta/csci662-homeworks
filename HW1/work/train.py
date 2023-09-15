import argparse
from naivebayes import *
from logisticregression import *


def get_arguments():
    parser = argparse.ArgumentParser(description="Text Classifier Trainer")
    parser.add_argument("-m", help="type of model to be trained: naivebayes, perceptron")
    parser.add_argument("-i", help="path of the input file where training file is in the form <text>TAB<label>")
    parser.add_argument("-o", help="path of the file where the model is saved") # Respect the naming convention for the model: make sure to name it {nb, perceptron}.{4dim, authors, odiya, products}.model for your best models in your workplace otherwise the grading script will fail

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if args.m == "naivebayes":
        # Optimized hyperparameters for each dataset
        if "questions.model" in args.o:
            threshold = 9
        elif "odiya.model" in args.o:
            threshold = 2
        elif "products.model" in args.o:
            threshold = 2
        else:
            threshold = 0
        model = NaiveBayes(model_file=args.o, threshold=threshold)
    elif args.m == "logreg":
        # Optimized hyperparameters for each dataset
        if "questions.model" in args.o:
            learning_rate=0.15
            epochs=500
            threshold=0
            max_features=150
            batch_size=32
        elif "odiya.model" in args.o:
            learning_rate=0.01
            epochs=1000
            threshold=10
            max_features=1000
            batch_size=256
        elif "products.model" in args.o:
            learning_rate=0.95
            epochs=100
            threshold=2
            max_features=1000
            batch_size=256
        else:
            learning_rate=0.35
            epochs=500
            threshold=5
            max_features=2000
            batch_size=64
        model = LogisticRegression(model_file=args.o,
                                   learning_rate=learning_rate,
                                   epochs=epochs,
                                   threshold=threshold,
                                   max_features=max_features,
                                   batch_size=batch_size)

    else:
        pass
    #     ## TODO Add any other models you wish to train
    #     model = None

    model = model.train(input_file=args.i)