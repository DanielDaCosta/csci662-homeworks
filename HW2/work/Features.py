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

    def __init__(self, input_file, embedding_file, threshold=0, max_features=None):
        super(Features_FeedForward, self).__init__(input_file)
        self.embedding_matrix = self.read_embedding_file(embedding_file) # Need to save EmbeddingMatrix values for inference
        self.threshold = threshold
        self.max_features = max_features
        self.vocabulary = None
        self.word2index = None
        self.index2word = None
        self.idf = None # Need to save IDF values for inference

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
    
    def create_vocabulary(self, tokenized_text, threshold, max_features=None):
        """Creat vocabulary from training set, considering only words
        that have an occurence > threshold.
        """
        # Append everything together in a dictionary
        flattened_list = [item for sublist in tokenized_text for item in sublist]
        flattened_list_count = Counter(flattened_list)

        # Sort the dictionary by values in descending order
        flattened_list_count = dict(sorted(flattened_list_count.items(), key=lambda item: item[1], reverse=True))

        # Considering only words that have an occurence > threshold.
        flattened_list_count_filter = {word:count for word, count in flattened_list_count.items() if count > threshold}

        # Limit the size of the vocabulary based on max_features
        if max_features:
            flattened_list_count_filter = dict(islice(flattened_list_count_filter.items(), max_features-1))

        # Add to vocabulary the Out-of-Vocabulary token
        return list(flattened_list_count_filter.keys()) + ['UNK']
    
    def tf_idf(self, tokenized_text):
        """Term frequency-inverse document frequency
        """
        # Create Vocabulary
        self.vocabulary = self.create_vocabulary(tokenized_text, self.threshold, self.max_features)
        self.word2index = {word: i for i, word in enumerate(self.vocabulary, start=0)}
        self.index2word = {i: word for i, word in enumerate(self.vocabulary, start=0)}

        size_vocabulary = len(self.vocabulary)
        n_documents = len(tokenized_text)
        tf_array = np.zeros((n_documents, size_vocabulary))
        idf_array = np.zeros(size_vocabulary) # Inverse Document Frequency
        words_per_document = np.zeros(n_documents)
        # Compute Term-Frequency
        for d_i, sentence in enumerate(tokenized_text, start=0):
            words_in_document = []
            for word in sentence:

                index_word = self.word2index.get(word)
                
                if word in self.word2index.keys():
                    tf_array[d_i][index_word] += 1
                    words_per_document[d_i] += 1
                    # Inverse Document Frequency
                    if word not in words_in_document: # does not count repeated words in the same document
                        words_in_document.append(word) 
                        idf_array[index_word] += 1 # number of documents containing the term
        tf = (tf_array + 1)/(words_per_document.reshape(-1, 1) + 1)
        # Smoothing: to avoid division by zero errors and to ensure that terms with zero document
        # frequency still get a non-zero IDF score
        idf = np.log((n_documents + 1)/(idf_array + 1)) + 1 # Smoothing

        self.idf = idf
        tf_idf = tf*idf
        return tf_idf # Shape (n_documents, vocabulary)
    
    def sort_by_tfidf(self, tfidf_matrix, max_seq_length):
        """Sort input documents based on tf*idf score.
        Return top "max_seq_length" words
        :param: tfidf_matrix
        :param: max_seq_length
        :return: sentences ordered by TF-IDF score
        """
        
        # Indices of sorted matrix in descending order
        indices = np.argsort(-tfidf_matrix, axis=1)
        tfidf_matrix_sorted = []

        # Create sorted matrix
        for i in range(tfidf_matrix.shape[0]):
            # sentence in orderd version
            tmp = [self.index2word[index] for index in indices[i][:max_seq_length]]
            tfidf_matrix_sorted.append(tmp)
    
        return tfidf_matrix_sorted
    
    def get_features_tfidf(self, tokenized_sentence, idf_array):
        """Convert sentence to TF-IDF space
        """
        size_vocabulary = len(self.vocabulary)
        tf_array = np.zeros(size_vocabulary)
        words_per_document = 0
        # Compute Term-Frequency
        words_in_document = []
        for word in tokenized_sentence:
            index_word = self.word2index.get(word)
            if word in self.word2index.keys():
                tf_array[index_word] += 1
                words_per_document += 1
        tf = (tf_array + 1)/(words_per_document+1) # with smoothinf
        return tf*idf_array
    
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