""" 
    Basic feature extractor
"""
from operator import methodcaller
import string
from collections import Counter, defaultdict
import numpy as np

def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.lower().split()

class Features:

    def __init__(self, data_file, has_label=True):
        with open(data_file) as file:
            data = file.read().splitlines()

        # If dataset has label; break into 
        if has_label:
            data_split = map(methodcaller("rsplit", "\t", 1), data)
            texts, self.labels = map(list, zip(*data_split))
            self.labelset = list(set(self.labels))
        else:
            texts = data
            self.labels = None
            self.labelset = None

        self.tokenized_text = [tokenize(text) for text in texts]



    @classmethod 
    def get_features(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features 
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features 
        pass

########################
# Naive Bayes Features #
########################

class Features_NB(Features):

    def __init__(self, model_file, has_label=True):
        super(Features_NB, self).__init__(model_file, has_label)
        self.vocabulary = self.create_vocabulary(self.tokenized_text)
        self.size_vocab = len(self.vocabulary)
        self.feature_weights = self.generate_feature_weights(laplace_smoothing=True)

    def read_input_file(self, input_file):

        with open(input_file) as file:
            data = file.read().splitlines()

        texts = data

        tokenized_text = [tokenize(text) for text in texts]
        return tokenized_text

    def count_frequency_word_label(self, sentences, labels):
        
        """
        :param sentences (list[list]): sentences tokenized
        :param labels (list): list of labels
        :return: count(c_j, w_i) refers to the count of word w_i in documents with label c_j
                _sum_{i=1}^{V}{count(c_j, w_i)} sum of the counts of each word in our vocabulary in class c_j 
                 count(c_j) refers to the count of label c_j 
        """
        count_word_label = []
        count_words_per_label = defaultdict(int)
        for sentence, label in zip(sentences, labels):
            for token in sentence:
                count_word_label.append((token, label))
                count_words_per_label[label] += 1
            
        # count_word_label = [(token, label) for sentence, label in zip(sentences, labels) for token in sentence]
        count_label = Counter(labels)
        return Counter(count_word_label), count_words_per_label, count_label

    def create_vocabulary(self, tokenized_text):

        # Append everything together in a dictionary
        flattened_list = [item for sublist in tokenized_text for item in sublist]
        flattened_list_count = Counter(flattened_list)
        vocabulary = list(flattened_list_count.keys())
        return vocabulary
    
    def generate_feature_weights(self, laplace_smoothing=True):
        # Vocabulary
        size_vocab = len(self.vocabulary)

        # Maximum Likelihood Estimates
        count_word, count_words_label, count_label = self.count_frequency_word_label(self.tokenized_text, self.labels)

        # Feature Weights
        feature_weights = defaultdict()
        feature_weights["count_word"] = count_word
        feature_weights["count_words_label"] = count_words_label
        feature_weights["count_label"] = count_label

        return feature_weights

    @classmethod 
    def get_features(cls, tokenized, target_label, model):

        # Compute log(P(w_i|c_j)
        features = []
        total_prob = 0.0
        for word in tokenized:
            word_label = (word, target_label)
            # Laplace Smoothing
            prob = (model.feature_weights["count_word"][word_label] + 1)/(model.feature_weights["count_words_label"][target_label] + model.size_vocab)
            prob_log = np.log(prob)
            total_prob += prob_log # save total

        # add prior probability
        n_documents = len(model.tokenized_text)
        total_prob += np.log(model.feature_weights["count_label"][target_label]/n_documents)
        return total_prob
