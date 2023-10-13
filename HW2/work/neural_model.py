from model import Model
from nn_layers import FeedForwardNetwork
import numpy as np
from Features import Features_FeedForward
import torch
import torch.optim as optim
import torch.nn as nn

class NeuralModel(Model):
    def __init__(self, embeddingfile,
                 max_seq_length,
                 hidden_units, minibatch_size,
                 learning_rate,
                 epochs,
                 hidden_units_other_layers=[],
                 tfidf=False,
                 max_features=None,
                 threshold=0,
                 momentum=0):
        '''
        :param embeddingfile: word embedding file
        :param hidden_units: number of hidden units
        :param minibatch_size: mini-batch size
        :param learning_rate: learning_rate: learning
        :param epochs: number of epochs to train for
        :param hidden_units_other_layers (list): number of hidden units in each layer
        :param tfidf: Enable TF-IDF ranking
        :param threshold: TF-IDF Vocabulary size
        :param momentum: TF-IDF Minimum word frequency required
        '''
        # self.network = FeedForwardNetwork()
        self.embeddingfile = embeddingfile
        self.embedding_dim = None
        self.max_seq_length = max_seq_length

        self.hidden_units = [hidden_units] +  hidden_units_other_layers if len(hidden_units_other_layers) > 0 else [hidden_units]
        # self.hidden_units = hidden_units if type(hidden_units) == list else [hidden_units] # list or int
        self.n_hidden_layers = len(self.hidden_units)
        self.weights = [None]*(self.n_hidden_layers + 1)
        self.bias = [None]*(self.n_hidden_layers + 1)


        self.Y_to_categorical = None
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.features_ff_class = None
        self.learning_rate = learning_rate
        self.loss = {}
        # TF-IDF Sorting
        self.tfidf = tfidf # enable sorting by tf-idf score
        self.max_features = max_features
        self.threshold = threshold
        # Momentum
        self.momentum = momentum
    
    def initialize_weights(self, n_inputs, n_output):
        # weights = np.zeros((n_inputs, n_output))
        # bias = np.zeros(n_output)
        # np.random.seed(0)
        weights = np.random.rand(n_inputs, n_output)
        bias = np.random.rand(n_output)
        return weights, bias
    
    def relu_function(self, A):
        '''A = x*W + b

        :return: Z = relut(x*A+b)
        '''
        return np.maximum(0, A)
    
    def relu_derivative(self, A):
        return np.where(A > 0, 1, 0)

    def cross_entropy_loss(self, S, target):
        """Calculate the cross-entropy
        L = -1/n*_sum_{i=0}^{n}{y_i*log(s_i)} 
        y label is a vector containing K classes where yc = 1 if c is the correct class and the remaining elements will be 0.

        :param S[num_documents, num_labels]: probabilities of features after softmax
        :target [num_documents, num_labels]: target one hot encoded
        """
        epsilon = 1e-15
        S = np.maximum(epsilon, S)
        S = np.minimum(1 - epsilon, S)
        return -np.mean(np.log(S)*target)

    def softmax(self, Z):
        """Softmax function: normalizing logit scores
        :param Z([num_documents, num_labels])
        :return e^Z/sum_{i=0}^{k}{e^{Z}}
        """
        return np.exp(Z - np.max(Z, axis=1, keepdims=True))/np.sum(np.exp(Z - np.max(Z, axis=1, keepdims=True)), axis=1, keepdims=True)
    
    def OneHot(self, targets, num_labels):
        """Convert arrary of targets to One Hot 
        :param targets([num_documents,])
        :param num_labels(int)
        :return Y[num_documents, num_labels]
        """
        Y_onehot = np.zeros((len(targets), num_labels))
        Y_onehot[np.arange(len(targets)), targets] = 1
        return Y_onehot
    
    def predict(self, X):
        """Return prediction of X with the categorical values]
        """
        # z[num_documents, num_labels] = X[num_documents, num_features]*W[num_features, num_labels] + bias[num_labels]
        Z_i = np.dot(X, self.weights[0]) + self.bias[0]
        A_i = self.relu_function(Z_i)
        i = 0
        if self.n_hidden_layers > 1:
            for i in range(self.n_hidden_layers-1):
                Z_i = np.dot(A_i, self.weights[i+1]) + self.bias[i+1]
                A_i = self.relu_function(Z_i)

            i = i + 1
        Z_i = np.dot(A_i, self.weights[i+1]) + self.bias[i+1]
        O = self.softmax(Z_i)

        # Rows with highest probability
        S_max = np.argmax(O, axis=1)

        return S_max
    
    def convert_to_embeddings(self, sentence):
        '''Convert sentence to embeddings
        '''
        emb = self.features_ff_class.get_features(sentence)
            # try:
        if emb: # if there is a word
            emb_concat = np.concatenate(emb, axis=0)
        else:
            emb_concat = []
        # If you need padding words (i.e., your input is too short), use a vector of zeroes
        if len(emb) < self.max_seq_length:
            # Missing words
            words_missing = self.max_seq_length - len(emb)
            # print(words_missing)
            emb_concat = np.pad(emb_concat, (0, words_missing*self.embedding_dim), 'constant')
        return emb_concat

    
    def train(self, input_file, verbose=False):

        # Read dataset and create vocabulary
        features_ff_class = Features_FeedForward(input_file, self.embeddingfile, threshold=self.threshold, max_features=self.max_features)
        self.features_ff_class = features_ff_class
        num_labels = len(features_ff_class.labelset)

        # Convert Y from categorical to integers values
        Y_mapping = {label: index for index, label in enumerate(np.unique(features_ff_class.labels))}
        self.Y_to_categorical = {index: label for label, index in Y_mapping.items()} # dictionary to convert back y's to categorical
        Y = [Y_mapping[y] for y in features_ff_class.labels]
        # Convert to OneHot for computing Loss
        Y_onehot = self.OneHot(Y, num_labels)

        # Get embedding dim
        self.embedding_dim = list(features_ff_class.embedding_matrix.values())[0].shape[0]

        # Number of sentences
        sample_size = len(features_ff_class.tokenized_text)

        # X_train: shape: 50f or 300f-dim × features (u)
        n_inputs = self.max_seq_length*self.embedding_dim # number of features
        X_train = np.zeros((sample_size, n_inputs))

        if self.tfidf: # Truncate input to the max sequence length sorted by TF-IDF
            tf_idf = features_ff_class.tf_idf(features_ff_class.tokenized_text)
            trunc_tokenized_text = features_ff_class.sort_by_tfidf(
                tf_idf,
                self.max_seq_length
            )
        else:
            # Truncate input to the max sequence length
            trunc_tokenized_text = features_ff_class.adjust_max_seq_length(
                features_ff_class.tokenized_text,
                self.max_seq_length
            )
        # Convert to embeddings with zero-padding
        for i, sentence in enumerate(trunc_tokenized_text):
            sentence_emb = self.convert_to_embeddings(sentence)
            X_train[i] = sentence_emb

        minibatch_size = self.minibatch_size

        # Initialize Weights
        # Create W_a and b_a
        # W_0[n_documents, hidden_units (u)]
        # b_0[hidden_units (u)]

        list_of_sizes = [n_inputs] + self.hidden_units + [num_labels]
        for i in range(self.n_hidden_layers + 1):
            weights, bias = self.initialize_weights(list_of_sizes[i], list_of_sizes[i+1])
            self.weights[i] = weights
            self.bias[i] = bias

        # Initilze Momentum weights
        prev_dW_i = [0] * (self.n_hidden_layers + 1)
        prev_db_i = [0] * (self.n_hidden_layers + 1)

        # Permutate the dataset to increase randomness
        np.random.seed(0)
        permutation = np.random.permutation(sample_size)
        # X_train[n_documents, n_features]
        X_permutation = X_train[permutation]
        Y_permutation_onehot = Y_onehot[permutation]

        for n_epoch in range(self.epochs):
            # Mini-batch_size Implementation
            mini_batch_loss = []
            for j in range(0, sample_size, minibatch_size):
                X_mini_batch = X_permutation[j:j+minibatch_size]
                y_mini_batch = Y_permutation_onehot[j:j+minibatch_size]

                ##########################################################
                # ---------------------FORWARD PASS--------------------- #
                ##########################################################
                # List of outputs of each layer
                # A[0] -> Input Layer
                # A[.] => Hidden Layer
                # A[n] -> Ouput Layer
                A = [None]*(self.n_hidden_layers + 2) 
                Z = [None]*(self.n_hidden_layers + 2)
                # ---------------- Input Layer --------------- #
                A[0] = X_mini_batch
                Z[0] = X_mini_batch

                # ---------------- Hidden Layers --------------- #
                for i in range(self.n_hidden_layers):
                    # Z_i = np.dot(X_mini_batch, self.weights_i) + self.bias_i
                    # A_i = relu(Z_i)
                    Z_tmp = np.dot(A[i], self.weights[i]) + self.bias[i]
                    Z[i+1] = Z_tmp
                    A_tmp = self.relu_function(Z_tmp)
                    A[i+1] = A_tmp
                # ---------------- Hidden-to-Output Layer --------------- #

                i = i + 1
                # print(i)
                Z_output_layer = np.dot(A[i], self.weights[self.n_hidden_layers]) + self.bias[self.n_hidden_layers]
                Z[i+1] = Z_output_layer
                A_output_layer = self.softmax(Z_output_layer)
                A[i+1] = A_output_layer


                ##########################################################
                # -------------------BACKWARD PASS---------------------- #
                ##########################################################

                # Compute Gradients
                # List of output gradients of each layer
                # dZ[0] -> Input Layer
                # dZ[.] => Hidden Layer
                # dZ[n] -> Ouput Layer
                dZ = [None] * (self.n_hidden_layers + 1)

                # dW[previous_layer, next_layer]
                dW = [None] * (self.n_hidden_layers + 1)
                db = [None] * (self.n_hidden_layers + 1)
                dZ[-1] = A[-1] - y_mini_batch
                for i in range(self.n_hidden_layers, 0, -1):
                    dW[i] = (1/minibatch_size)*np.dot(A[i].T, dZ[i])
                    db[i] = (1/minibatch_size)*np.sum(dZ[i], axis=0, keepdims = True)
                    dZ[i-1] = np.dot(dZ[i], self.weights[i].T)*self.relu_derivative(Z[i])

                # print(dZ[i-1])
                dW[0] = (1/minibatch_size)*np.dot(X_mini_batch.T, dZ[i-1])
                db[0] = (1/minibatch_size)*np.sum(dZ[i-1], axis=0, keepdims = True)

                # Update Weights
                for i in range(self.n_hidden_layers + 1):
                    self.weights[i] = self.weights[i] - (self.learning_rate*dW[i] + self.momentum*prev_dW_i[i])
                    self.bias[i] = self.bias[i] - (self.learning_rate*db[i] + self.momentum*prev_db_i[i])
                    # Momentum
                    # Save previous gradients for Momentum
                    prev_dW_i[i] =  self.learning_rate*dW[i] + self.momentum*prev_dW_i[i]
                    prev_db_i[i] =  self.learning_rate*db[i] + self.momentum*prev_db_i[i]

                ########
                # Loss #
                ########
                # print(y_mini_batch)
                mini_batch_loss.append(self.cross_entropy_loss(A[-1], y_mini_batch))

            loss = np.mean(mini_batch_loss)
            self.loss[n_epoch] = loss
            if verbose:
                print(f"Epoch: {n_epoch+1} - Loss: {loss}")

    def classify(self, input_file):
        # Read Input File
        tokenized_text = self.features_ff_class.read_inference_file(input_file)

        if self.tfidf:
            tf_idf_inference = []
            # Get features from inference file
            for sentence in tokenized_text:
                # Transform dataset to TF-IDF space
                # Return features with format (1, size_vocabulary)
                X_sentence = self.features_ff_class.get_features_tfidf(sentence, self.features_ff_class.idf)
                tf_idf_inference.append(X_sentence)
            tf_idf_inference = np.stack(tf_idf_inference)
            trunc_tokenized_text = self.features_ff_class.sort_by_tfidf(
                tf_idf_inference,
                self.max_seq_length
            )
        else:
            # Truncate input to the max sequence length
            trunc_tokenized_text = self.features_ff_class.adjust_max_seq_length(
                tokenized_text,
                self.max_seq_length
            )

        X_test = []
        # Convert to embeddings with zero padding
        for i, sentence in enumerate(trunc_tokenized_text):
            sentence_emb = self.convert_to_embeddings(sentence)
            X_test.append(sentence_emb)
        X_test = np.vstack(X_test)

        # Make Prediction
        y_test = self.predict(X_test)
        preds_label = []
        for y in y_test:
            tmp = self.Y_to_categorical[y]
            preds_label.append(tmp)
        
        return preds_label


#################
# PyTorch Model #
#################

class NeuralNetworkTorch(nn.Module):
    def __init__(self, input_dim, hidden_units, n_labels):
        super(NeuralNetworkTorch, self).__init__()
        self.weights_1 = nn.Linear(input_dim, hidden_units)
        self.relu = nn.ReLU()
        self.weights_2 = nn.Linear(hidden_units, n_labels)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the output dimension

    def forward(self, x):
        # ---------------- Input-to-Hidden Layer --------------- #
        x = self.weights_1(x)
        x = self.relu(x)
        # ---------------- Hidden-to-Output Layer --------------- #
        x = self.weights_2(x)
        x = self.softmax(x)
        return x

class NeuralModel_Torch(Model):
    def __init__(self, embeddingfile, max_seq_length, hidden_units, minibatch_size, learning_rate, epochs, adam=False): 
        self.embeddingfile = embeddingfile
        self.embedding_dim = None
        self.max_seq_length = max_seq_length
        self.hidden_units = hidden_units
        # Layers
        self.model_torch = None
        self.Y_to_categorical = None
        self.criterion = nn.CrossEntropyLoss()
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.features_ff_class = None
        self.learning_rate = learning_rate
        self.adam = adam
        self.loss = {}
        
    def convert_to_embeddings(self, sentence):
        '''Convert sentence to embeddings
        '''
        emb = self.features_ff_class.get_features(sentence)
            # try:
        if emb: # if there is a word
            emb_concat = np.concatenate(emb, axis=0)
        else:
            emb_concat = []
        # If you need padding words (i.e., your input is too short), use a vector of zeroes
        if len(emb) < self.max_seq_length:
            # Missing words
            words_missing = self.max_seq_length - len(emb)
            # print(words_missing)
            emb_concat = np.pad(emb_concat, (0, words_missing*self.embedding_dim), 'constant')
        return emb_concat

    
    def train(self, input_file, verbose=False):

        # Read dataset and create vocabulary
        features_ff_class = Features_FeedForward(input_file, self.embeddingfile)
        self.features_ff_class = features_ff_class
        num_labels = len(features_ff_class.labelset)

        # Convert Y from categorical to integers values
        Y_mapping = {label: index for index, label in enumerate(np.unique(features_ff_class.labels))}
        self.Y_to_categorical = {index: label for label, index in Y_mapping.items()} # dictionary to convert back y's to categorical
        Y = [Y_mapping[y] for y in features_ff_class.labels]
        Y = np.array(Y)
        # Convert to OneHot for computing Loss
        # Y_onehot = self.OneHot(Y, num_labels)

        # Get embedding dim
        self.embedding_dim = list(features_ff_class.embedding_matrix.values())[0].shape[0]

        # Number of sentences
        sample_size = len(features_ff_class.tokenized_text)

        # X_train: shape: 50f or 300f-dim × features (u)
        n_inputs = self.max_seq_length*self.embedding_dim # number of features
        X_train = np.zeros((sample_size, n_inputs))

        # Truncate input to the max sequence length
        trunc_tokenized_text = features_ff_class.adjust_max_seq_length(
            features_ff_class.tokenized_text,
            self.max_seq_length
        )

        # Convert to embeddings with zero-padding
        for i, sentence in enumerate(trunc_tokenized_text):
            sentence_emb = self.convert_to_embeddings(sentence)
            X_train[i] = sentence_emb

        minibatch_size = self.minibatch_size

        # Initialize Torch Model
        self.model_torch = NeuralNetworkTorch(n_inputs, self.hidden_units, num_labels)
        # Optimzer
        if self.adam:  
            optimizer = optim.Adam(self.model_torch.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.SGD(self.model_torch.parameters(), lr=self.learning_rate)

        #################
        # Torch Tensors #
        #################
        # Permutate the dataset to increase randomness
        np.random.seed(0)
        permutation = np.random.permutation(sample_size)
        # X_train[n_documents, n_features]
        X_permutation = X_train[permutation]
        Y_permutation = Y[permutation]


        # Torch Tensors
        X_permutation = torch.tensor(X_permutation, dtype=torch.float32)
        Y_permutation = torch.tensor(Y_permutation)

        for i in range(self.epochs):
            # Mini-batch_size Implementation
            mini_batch_loss = []
            for j in range(0, sample_size, minibatch_size):
                X_mini_batch = X_permutation[j:j+minibatch_size]
                y_mini_batch = Y_permutation[j:j+minibatch_size]

                ##########################################################
                # ---------------------FORWARD PASS--------------------- #
                ##########################################################
                optimizer.zero_grad()
                outputs = self.model_torch.forward(X_mini_batch)

                loss = self.criterion(outputs, y_mini_batch)

                ##########################################################
                # -------------------BACKWARD PASS---------------------- #
                ##########################################################
   
                loss.backward()
                optimizer.step()

                mini_batch_loss.append(loss.item())

            
            self.loss[i] = np.mean(mini_batch_loss)
            if verbose:
                print(f"Epoch: {i+1} - Loss: {self.loss[i]}")

    def classify(self, input_file):
        # Read Input File
        tokenized_text = self.features_ff_class.read_inference_file(input_file)

        # Truncate input to the max sequence length
        trunc_tokenized_text = self.features_ff_class.adjust_max_seq_length(
            tokenized_text,
            self.max_seq_length
        )
        X_test = []
        # Convert to embeddings with zero padding
        for i, sentence in enumerate(trunc_tokenized_text):
            sentence_emb = self.convert_to_embeddings(sentence)
            X_test.append(sentence_emb)
        X_test = np.vstack(X_test)

        # Convert to tensor
        X_test = torch.tensor(X_test, dtype=torch.float32)

        # Make Prediction
        preds_label = []
        with torch.no_grad():
            predicted = self.model_torch(X_test)
            _, y_test = torch.max(predicted, 1)
            for y in y_test:
                tmp = self.Y_to_categorical[y.item()] # Convert to original class
                preds_label.append(tmp)
        
        return preds_label