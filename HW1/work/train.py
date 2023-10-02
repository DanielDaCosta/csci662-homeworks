import argparse
from naivebayes import *
from logisticregression import *
from logisticregression_word2vec_extramile import *
from naivebayes_tfidf_extramile import *
from gensim.models import KeyedVectors


def get_arguments():
    parser = argparse.ArgumentParser(description="Text Classifier Trainer")
    parser.add_argument("-m", help="type of model to be trained: naivebayes, perceptron")
    parser.add_argument("-i", help="path of the input file where training file is in the form <text>TAB<label>")
    parser.add_argument("-o", help="path of the file where the model is saved") # Respect the naming convention for the model: make sure to name it {nb, perceptron}.{4dim, authors, odiya, products}.model for your best models in your workplace otherwise the grading script will fail
    # Extra Mile Only
    parser.add_argument("-e", help="Word2Vec embedding")

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
    ##############
    # Extra Mile #
    ##############
    elif args.m == "logreg_word2vec":
        if "questions.model" in args.o:
            learning_rate=0.15
            epochs=500
            batch_size=32
        elif "odiya.model" in args.o:
            learning_rate=0.01
            epochs=1000
            batch_size=256
        elif "products.model" in args.o:
            learning_rate=0.95
            epochs=100
            batch_size=256
        else:
            learning_rate=0.35
            epochs=500
            threshold=5
            max_features=2000
            batch_size=64
        if not args.e:
            print("Please provide the path to word2vec.wordvectors embedding")
            exit(1)
        with open(args.e, "rb") as file:
            word2vec_embedding = pickle.load(file)
        model = LogisticRegressionWord2Vec(model_file=args.o,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            batch_size=batch_size,
                            embedding_matrix=word2vec_embedding)
    elif args.m == "naivebayes_tfidf":
        # Optimized hyperparameters for each dataset
        if "questions.model" in args.o:
            threshold = 9
        elif "odiya.model" in args.o:
            threshold = 2
        elif "products.model" in args.o:
            threshold = 2
        else:
            threshold = 0
        model = NaiveBayes_TF_IDF(model_file=args.o, threshold=threshold)


    model = model.train(input_file=args.i)