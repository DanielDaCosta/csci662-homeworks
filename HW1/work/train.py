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
            earning_rate=0.005
            epochs=400
            threshold=0
            max_features=100
        elif "odiya.model" in args.o:
            learning_rate=0.000001
            epochs=1000
            threshold=10
            max_features=1000
        elif "products.model" in args.o:
            learning_rate=0.9
            epochs=1000
            threshold=1
            max_features=500
        else:
            learning_rate=0.2
            epochs=200
            threshold=1
            max_features=100
        model = LogisticRegression(model_file=args.o,
                                   learning_rate=learning_rate,
                                   epochs=epochs,
                                   threshold=threshold,
                                   max_features=max_features)

    else:
        pass
    #     ## TODO Add any other models you wish to train
    #     model = None

    model = model.train(input_file=args.i)