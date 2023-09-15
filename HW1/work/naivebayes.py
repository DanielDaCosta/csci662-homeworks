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
import math

class NaiveBayes(Model):

    def __init__(self, model_file, threshold=0):
        super(NaiveBayes, self).__init__(model_file)
        self.threshold = threshold

    def __count_frequency_word_label(self, sentences, labels):
        
        """
        :param sentences (list[list]): sentences tokenized
        :param labels (list): list of labels
        :return: count(c_j, w_i) refers to the count of word w_i in documents with label c_j
                _sum_{i=1}^{V}{count(c_j, w_i)} sum of the counts of each word in our vocabulary in class c_j 
                 count(c_j) refers to the count of label c_j 
        """
        count_word_label = []
        # count_words_per_label = defaultdict(int)
        for sentence, label in zip(sentences, labels):
            for token in sentence:
                count_word_label.append((token, label))
                # count_words_per_label[label] += 1
            
        # count_word_label = [(token, label) for sentence, label in zip(sentences, labels) for token in sentence]
        count_label = Counter(labels)
        count_word_label = Counter(count_word_label)

        count_words_per_label = defaultdict(int)
        for (word, label), count in count_word_label.items():
            count_words_per_label[label] += count + 1 # Add laplace smoothing

        return count_word_label, count_words_per_label, count_label
    
    def __compute_feature_weights(self, count_word_label, count_words_per_label, count_label, size_vocabulary, alpha=0):
        """
        :param alpha (int): Hyperparemeter alpha for Laplace Smoothing
        """
        feature_weights = defaultdict(dict)
        for word, label in count_word_label.keys():
            # Maximum Likelihood Estimates
            # tmp = math.log((count_word_label[(word, label)] + alpha)/(size_vocabulary*alpha + count_words_per_label[label]))
            tmp = math.log((count_word_label[(word, label)])/(count_words_per_label[label]))
            feature_weights[label][word] = tmp

        # Include Probability of each label: 
        total_documents = sum(count_label.values())
        for label in count_label.keys():
            probability_label_name = "prob_mu"
            feature_weights[label][probability_label_name] = math.log(count_label[label]/total_documents)
            probability_label_name = "laplace_smoothing" # Weights for token with count(y,j) = 0 
            feature_weights[label][probability_label_name] = math.log(1/(count_words_per_label[label]))
        return feature_weights

    
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """

        # Instanciate Features_NB class:
        #   - Create Vocabulary
        features_naive_bayes = Features_NB(input_file, self.threshold)

        # Replace words that are not in vocabulary with OOV (Out-of-Vocabulary)
        # token
        updated_text = []
        labels = features_naive_bayes.labels
        for sentence in features_naive_bayes.tokenized_text:
            tmp = features_naive_bayes.replace_unknown_word_with_oov(sentence)
            updated_text.append(tmp)            

        # Compute Feature Weights
        count_word_label, count_words_per_label, count_label = self.__count_frequency_word_label(updated_text, labels)
        feature_weights = self.__compute_feature_weights(count_word_label, count_words_per_label, count_label, len(features_naive_bayes.vocabulary))

        # Build Model
        nb_model = {
            "feature_weights": feature_weights,
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
        Feature_NB_class = model["Feature"]

        # Read Input File
        tokenized_text = Feature_NB_class.read_inference_file(input_file)

        preds = []

        # Choosing the label y which maximizes log p(x, y; μ, φ):
        class_predictions_list = []
        for sentence in tokenized_text:
            sentence_features = Feature_NB_class.get_features(sentence, model)
            # print("Sentence Characters: ", len(sentence_features.keys()))
            class_predictions = defaultdict()

            for label in Feature_NB_class.labelset:
                feature_weights_y = feature_weights[label]
                # Compute Inner Product: feature_weights*feature_vector
                total_sum = 0 
                for key in sentence_features.keys():
                    if key in feature_weights_y.keys():
                        tmp = sentence_features[key] * feature_weights_y[key]
                    else:

                        tmp = sentence_features[key] * feature_weights_y['laplace_smoothing']
                    total_sum += tmp

                class_predictions[label] = total_sum
            # Find the class with the highest value
            class_predictions_list.append(class_predictions) # for debugging
            class_with_highest_value = max(class_predictions, key=lambda k: class_predictions[k])
            preds.append(class_with_highest_value)

        return preds