"""
NaiveBayes is a generative classifier based on the Naive assumption that features are independent from each other
P(w1, w2, ..., wn|y) = P(w1|y) P(w2|y) ... P(wn|y)
Thus argmax_{y} (P(y|w1,w2, ... wn)) can be modeled as argmax_{y} P(w1|y) P(w2|y) ... P(wn|y) P(y) using Bayes Rule
and P(w1, w2, ... ,wn) is constant with respect to argmax_{y} 
Please refer to lecture notes Chapter 4 for more details
"""
from Features import Features_NB
from Model import *
from collections import Counter, defaultdict
import numpy as np
class NaiveBayes(Model):

    def __init__(self, model_file):
        super(NaiveBayes, self).__init__(model_file)

    
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """

        features_naive_bayes = Features_NB(input_file, True)
        self.save_model(features_naive_bayes)
    
        

    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """ 

        # Read Input File
        tokenized_text = model.read_input_file(input_file)

        preds = []
        for sentence in tokenized_text:
            class_predictions = defaultdict()
            for label in set(model.labels):
                class_predictions[label] = model.get_features(sentence, label, model)
            # Find the class with the highest value
            class_with_highest_value = max(class_predictions, key=lambda k: class_predictions[k])
            preds.append(class_with_highest_value)
        
        return preds