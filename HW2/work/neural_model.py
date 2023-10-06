from model import Model
from nn_layers import FeedForwardNetwork
import numpy as np
from Features import Features_FeedForward

class NeuralModel(Model):
    def __init__(self, embeddingfile, max_seq_length, hidden_units, minibatch_size, learning_rate, epochs): 
        # self.network = FeedForwardNetwork()
        self.embeddingfile = embeddingfile
        self.embedding_dim = None
        self.max_seq_length = max_seq_length
        self.hidden_units = hidden_units
        self.weights_1 = None
        self.bias_1 = None
        self.weights_2 = None
        self.bias_2 = None
        self.Y_to_categorical = None
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.features_ff_class = None
        self.learning_rate = learning_rate
        self.loss = {}
    
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
        A = np.dot(X, self.weights_1) + self.bias_1
        h = self.relu_function(A)

        A_2 = np.dot(h, self.weights_2) + self.bias_2

        O = self.softmax(A_2)

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
        features_ff_class = Features_FeedForward(input_file, self.embeddingfile)
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

        # Initialize Wieghts
        # Create W_a and b_a
        # W_a[n_documents, hidden_units (u)]
        # b_a[hidden_units (u)]
        W_1, b_1 = self.initialize_weights(n_inputs, self.hidden_units)
        # Create Wb and b_b
        # W_b[hidden_units (u), num_labels (d)]
        # b_b[num_labels]
        W_2, b_2 = self.initialize_weights(self.hidden_units, num_labels)

        # Permutate the dataset to increase randomness
        np.random.seed(0)
        permutation = np.random.permutation(sample_size)
        # X_train[n_documents, n_features]
        X_permutation = X_train[permutation]
        Y_permutation_onehot = Y_onehot[permutation]

        self.weights_1 = W_1
        self.bias_1 = b_1
        self.weights_2 = W_2
        self.bias_2 = b_2
        for i in range(self.epochs):
            # Mini-batch_size Implementation
            mini_batch_loss = []
            for j in range(0, sample_size, minibatch_size):
                X_mini_batch = X_permutation[j:j+minibatch_size]
                y_mini_batch = Y_permutation_onehot[j:j+minibatch_size]

                ##########################################################
                # ---------------------FORWARD PASS--------------------- #
                ##########################################################
            
                # ---------------- Input-to-Hidden Layer --------------- #
                # Z1 = W_a*X + b_a
                # Z1[n_documents, hidden_units (u)]
                Z_1 = np.dot(X_mini_batch, self.weights_1) + self.bias_1
                # Hidden Unit
                # h = relu(A)
                # h[n_documents, hidden_units (u)]
                A_1 = self.relu_function(Z_1)

                # ---------------- Hidden-to-Output Layer --------------- #
                #  = W_b*h + b_b
                # A_2[n_documents, num_labels (d)]
                Z_2 = np.dot(A_1, self.weights_2) + self.bias_2
                # Output Layer
                # A_2 = softmax(Z_2)
                # A_2[n_documents, num_labels (d)]
                A_2 = self.softmax(Z_2)

                ##########################################################
                # -------------------BACKWARD PASS---------------------- #
                ##########################################################

                # Compute Gradients

                dZ_2 = A_2 - y_mini_batch # [n_documents, num_labels (d)]
                # np.dot(A_2, dZ_2) => (hidden_units, n_documents) X (n_documents, num_labels) = (hidden_units, num_labels)
                dW_2 = (1/minibatch_size)*np.dot(A_1.T, dZ_2)
                db_2 = (1/minibatch_size)*np.sum(dZ_2, axis=0, keepdims = True) # [num_labels]
                # np.dot(self.weights_b, dZ_2) => [n_documents, num_labels (d)] X [num_labels (d), hidden_units (u)] => [n_documents, hidden_units]
                dZ_1 = np.dot(dZ_2, self.weights_2.T)*self.relu_derivative(Z_1)
                # np.dot(X, dZ_1) => (features, n_documents) X (n_documents, hidden_units) = (hidden_units, num_labels)
                dW_1 = (1/minibatch_size)*np.dot(X_mini_batch.T, dZ_1)
                db_1 = (1/minibatch_size)*np.sum(dZ_1, axis=0, keepdims = True) # [hidden_units]

                # Update weights
                self.weights_1 = self.weights_1 - self.learning_rate*dW_1
                self.bias_1 = self.bias_1 - self.learning_rate*db_1
                self.weights_2 = self.weights_2 - self.learning_rate*dW_2
                self.bias_2 = self.bias_2 - self.learning_rate*db_2

                ########
                # Loss #
                ########
                mini_batch_loss.append(self.cross_entropy_loss(A_2, y_mini_batch))

            loss = np.mean(mini_batch_loss)
            self.loss[i] = loss
            if verbose:
                print(f"Epoch: {i+1} - Loss: {loss}")

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

        # Make Prediction
        y_test = self.predict(X_test)
        preds_label = []
        for y in y_test:
            tmp = self.Y_to_categorical[y]
            preds_label.append(tmp)
        
        return preds_label
