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
    return text.split()

class Features:

    def __init__(self, data_file):
        with open(data_file) as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, self.labels = map(list, zip(*data_split))

        self.tokenized_text = [tokenize(text) for text in texts]

        self.labelset = list(set(self.labels))

    @classmethod 
    def get_features(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features 
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features 
        pass

########################
# Naive Bayes Features #
########################

class Features_NB(Features):

    def __init__(self, model_file):
        super(Features_NB, self).__init__(model_file)
        self.vocabulary = self.create_vocabulary(self.tokenized_text)

    def read_inference_file(self, input_file):
        """Read inference file that is in the form: <text> i.e. a line
        of text that does not contain a tab.
        """
        with open(input_file) as file:
            data = file.read().splitlines()

        texts = data

        tokenized_text = [tokenize(text) for text in texts]
        return tokenized_text
    
    def create_vocabulary(self, tokenized_text, threshold=0):
        """Creat vocabulary from training set, considering only words
        that have an occurence > threshold.
        """
        # Append everything together in a dictionary
        flattened_list = [item for sublist in tokenized_text for item in sublist]
        flattened_list_count = Counter(flattened_list)

        # Considering only words that have an occurence > threshold.
        flattened_list_count_filter = [word for word, count in flattened_list_count.items() if count > threshold]

        return flattened_list_count_filter

    def replace_unknown_word_with_oov(self, tokenized_sentence):
        """Replace words that are not in vocabulary with OOV (Out-of-Vocabulary)
        token
        """
        updated_sentence = []
        for word in tokenized_sentence:
            if word not in self.vocabulary:
                updated_sentence.append('OOV')
            else:
                updated_sentence.append(word)
        return updated_sentence
        
    def get_features(self, tokenized, model):
        """Bag-of-words: return column vector of word counts, including OOV (Out-of-Vocabulary) token, if present.
        Vector stores only non-zero values to improve performance
        """

        # Replace words that are not in vocabulary with OOV
        updated_text = model["Feature"].replace_unknown_word_with_oov(tokenized)

        bag_of_words = Counter(updated_text)
        # Include OffsetFeature "prob_mu" to 1; which allows to include the probability of the label
        # to the maximum likelihood estimation.

        bag_of_words["prob_mu"] = 1
        return bag_of_words
        