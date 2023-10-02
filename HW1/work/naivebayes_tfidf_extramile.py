"""
NaiveBayes is a generative classifier based on the Naive assumption that features are independent from each other
P(w1, w2, ..., wn|y) = P(w1|y) P(w2|y) ... P(wn|y)
Thus argmax_{y} (P(y|w1,w2, ... wn)) can be modeled as argmax_{y} P(w1|y) P(w2|y) ... P(wn|y) P(y) using Bayes Rule
and P(w1, w2, ... ,wn) is constant with respect to argmax_{y} 
Please refer to lecture notes Chapter 4 for more details
"""
from Features import Features_NB_TF_IDF
from Model import *
from collections import Counter, defaultdict
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from operator import methodcaller

class NaiveBayes_TF_IDF(Model):

    def __init__(self, model_file, threshold=None):
        super(NaiveBayes_TF_IDF, self).__init__(model_file)
        self.threshold = threshold # Minimum number of occurences of word
    
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        # Read dataset

        # train_dataset = pd.read_csv(input_file, sep='\t', header=None, names=['text', 'true_label'])

        with open(input_file) as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, labels = map(list, zip(*data_split))
        train_dataset = pd.DataFrame({'text': texts, 'true_label': labels})

        
        X_train = train_dataset['text']
        y_train = train_dataset['true_label']


        # Instanciate Features_NB_TF_IDF class:
        #   - Create TF-IDF Matrix
        features_naive_bayes = Features_NB_TF_IDF(input_file, self.threshold)
        
        X_train_tf = features_naive_bayes.get_features(X_train)


        # Train Model
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train_tf, y_train)

        # Build Model
        nb_model = {
            "NaiveBayes": naive_bayes,
            "feature_weights": features_naive_bayes.tfidf_vectorizer,
            "Feature": features_naive_bayes
        }
        
        self.save_model(nb_model)
        return nb_model
    
        

    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """ 
        feature_weights = model["feature_weights"]
        Feature_NB_TF_IDF_class = model["Feature"]
        NaiveBayes = model["NaiveBayes"]

        # Read dataset
        # test_dataset = pd.read_csv(input_file, sep='\t', header=None, names=['text'])

        with open(input_file) as file:
            data = file.read().splitlines()

        texts = data
        test_dataset = pd.DataFrame({'text': data})
        
        X_test = test_dataset['text'].values.astype('U')

        # return X_test

        X_test_tfidf = feature_weights.transform(X_test)

        # Make Prediction
        y_pred = NaiveBayes.predict(X_test_tfidf)

        # Convert to string for saving
        preds = [str(num) for num in y_pred]
        
        return preds