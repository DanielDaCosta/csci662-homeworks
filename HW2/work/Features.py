""" 
    Basic feature extractor
"""
from operator import methodcaller
import string
import re
from collections import Counter, defaultdict
import numpy as np
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def expand_contradictions(text):

    contraction_mapping = {
        "won't": "will not",
        "can't": "can not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
    }

    pattern = re.compile(r"\b(?:" + "|".join(re.escape(contraction) for contraction in contraction_mapping.keys()) + r")\b")
    text = pattern.sub(lambda x: contraction_mapping[x.group()], text)
    
    return text

def remove_digits_and_words_digits(text):
    # Define a regular expression pattern to match words containing digits
    pattern = r'\b\w*\d\w*\b'
    text_without_words_with_digits = re.sub(pattern, '', text)

    return text_without_words_with_digits

def remove_stop_words(text):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                  'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                  'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                  'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                  'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                  'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                  'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                  'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                  'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                  "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                  'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    text_clean = []
    for word in text:
        if word not in stop_words:
            text_clean.append(word)
    return text_clean


def tokenize(text, split=True):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    # re.sub('[^a-zA-Z]', '', dataset['Text'][i])

    # Text preprocessing techniques:
    # 1) Lowercase
    text = text.lower()

    # 2) Expand Contradictions
    text = expand_contradictions(text)

    # 3) Remove punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '' , text)

    # 4) Remove digits and words with digits
    text = remove_digits_and_words_digits(text)
    if split:
        text = text.split()

    # # 5) Remove Stop Words
    # if stop_words:
    # text = remove_stop_words(text)

    return text

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

class Features_FeedForward(Features):

    def __init__(self, input_file, embedding_file):
        super(Features_FeedForward, self).__init__(input_file)
        self.embedding_matrix = self.read_embedding_file(embedding_file) # Need to save EmbeddingMatrix values for inference

    def adjust_max_seq_length(self, tokenized_text, max_seq_length):
        """Adjust size of data input to the max sequence length
        :param tokenized_text: data input
        :param max_seq_length: the max sequence length
        :return list: truncated sentences
        """
        new_tokenized_text = []
        for sentence in tokenized_text:
            new_tokenized_text.append(sentence[:max_seq_length])
        return new_tokenized_text

        
    def read_embedding_file(self, embedding_file):
        '''Read embedding file

        :param embedding_file (str):
        :return: dict: embedding matrix
        '''

        embedding_matrix = dict()
        try: 
            with open(embedding_file, "r") as file:
                for line in file:
                    values = line.strip().split()
                    word = values[0]
                    word_embedding = np.array([float(emb) for emb in values[1:]])
                    embedding_matrix[word] = word_embedding
            return embedding_matrix
        except OSError as e:
            print("Embedding file " + embedding_file + " is not available, please input the right parth to the file.")

    def read_inference_file(self, input_file):
        """Read inference file that is in the form: <text> i.e. a line
        of text that does not contain a tab.
        """
        with open(input_file) as file:
            data = file.read().splitlines()

        texts = data

        tokenized_text = [tokenize(text) for text in texts]
        return tokenized_text
    
    def get_features(self, tokenized_sentence):
        """Convert sentence to word embeeding values.
        :param tokenized_sentence
        :return feature weights
        """
        sentence_embedding = []
        
        for word in tokenized_sentence:
            # get embedding of word if exists
            try:
                word_emb = self.embedding_matrix[word]
            except: # read UNK token embedding 
                word_emb = self.embedding_matrix["UNK"]
            sentence_embedding.append(word_emb)
        
        return sentence_embedding