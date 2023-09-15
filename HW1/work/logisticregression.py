"""
 Refer to Chapter 5 for more details on how to implement a LogisticRegression
"""
from Model import *
from Features import Features_LR
import numpy as np

class LogisticRegression(Model):
    def __init__(self, model_file, learning_rate=None, epochs=None, threshold=None, max_features=None):
        super(LogisticRegression, self).__init__(model_file)
        self.weights = None
        self.bias = None
        self.loss = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.Y_to_categorical = None # Map Y label to numerical
        self.threshold = threshold
        self.max_features = max_features

    def initialize_weights(self, num_features, num_labels):
        self.weights = np.zeros((num_features, num_labels))
        self.bias = np.zeros(num_labels)
        # np.random.seed(0)
        # self.weights = np.random.rand(num_features, num_labels)
        # self.bias = np.random.rand(num_labels)

    def softmax(self, Z):
        """Softmax function: normalizing logit scores
        :param Z([num_documents, num_labels])
        :return e^Z/sum_{i=0}^{k}{e^{Z}}
        """
        return np.exp(Z - np.max(Z, axis=1, keepdims=True))/np.sum(np.exp(Z), axis=1, keepdims=True)
    
    def sigmoid(self, Z):
        """Sigmoid function for binary classification

        :param Z([num_documents, num_labels])
        :return 1/(1+e^{-Z})
        """
        return 1/(1 + np.exp(-Z))

        
    def predict_prob(self, X, weights, bias, multinomial):
        """Return prediction of shape [num_documents, num_labels]
        """
        # z[num_documents, num_labels] = X[num_documents, num_features]*W[num_features, num_labels] + bias[num_labels]
        Z = np.dot(X, weights) + bias

        if multinomial:
            # Apply Softmax
            S = self.softmax(Z)
        else:
            # Apply Sigmoid
            S = self.sigmoid(Z)
        return S

    def cross_entropy_loss(self, S, target):
        """Calculate the cross-entropy
        L = -1/n*_sum_{i=0}^{n}{y_i*log(s_i)} 
        y label is a vector containing K classes where yc = 1 if c is the correct class and the remaining elements will be 0.

        :param S[num_documents, num_labels]: probabilities of features after softmax
        :target [num_documents, num_labels]: target one hot encoded
        """
        return -np.mean(np.log(S)*target)
    
    def binary_cross_entropy_loss(self, S, target):
        """Calculate Binary cross-entropy
        """
        return  -np.mean(target*(np.log(S)) + (1-target)*np.log(1-S))

    def OneHot(self, targets, num_labels):
        """Convert arrary of targets to One Hot 
        :param targets([num_documents,])
        :param num_labels(int)
        :return Y[num_documents, num_labels]
        """
        Y_onehot = np.zeros((len(targets), num_labels))
        Y_onehot[np.arange(len(targets)), targets] = 1
        return Y_onehot
    
    def predict(self, X, weights, bias, multinomial):
        """Return prediction of X with the categorical values]
        """
        # z[num_documents, num_labels] = X[num_documents, num_features]*W[num_features, num_labels] + bias[num_labels]
        Z = np.dot(X, weights) + bias

        if multinomial:
            # Apply Softmax
            S = self.softmax(Z)

            # Rows with highest probability
            S_max = np.argmax(S, axis=1)
        else:
            # Apply Sigmoid
            S = self.sigmoid(Z)
            print(S)
            # Rows with highest probability
            S_max = [1 if i > 0.5 else 0 for i in S]

        return S_max
    

    def train(self, input_file, verbose=False):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        # Read dataset and create vocabulary
        features_lr_class = Features_LR(input_file, self.threshold, self.max_features)

        # Replace words that are not in vocabulary with OOV (Out-of-Vocabulary)
        # token
        updated_text = []
        for sentence in features_lr_class.tokenized_text:
            tmp = features_lr_class.replace_unknown_word_with_oov(sentence)
            updated_text.append(tmp)

        # Transform dataset to TF-IDF space
        # Return features with format (n_documents, size_vocabulary)
        X = features_lr_class.tf_idf(updated_text)
        
        # Y
        Y_mapping = {label: index for index, label in enumerate(np.unique(features_lr_class.labels))}
        self.Y_to_categorical = {index: label for label, index in Y_mapping.items()} # dictionary to convert back y's to categorical
        Y = [Y_mapping[y] for y in features_lr_class.labels]

        # Initialize Weights
        sample_size = len(features_lr_class.tokenized_text)
        n_features = len(features_lr_class.vocabulary)
        num_labels = len(features_lr_class.labelset)


        # Check if it's multinomial or binary classification
        if num_labels == 2:
            multinomial = False
            num_labels = 1 # Only one column to reference 0 or 1
        else:
            multinomial = True

        self.initialize_weights(n_features, num_labels)

        # One Hot encoded Y
        if multinomial:
            Y_onehot = self.OneHot(Y, num_labels)
        else:
            Y_onehot = np.array(Y).reshape(-1, 1)

        for i in range(self.epochs):
            # Z = softmax(X*W + b)
            prob = self.predict_prob(X, self.weights, self.bias, multinomial)

            # break            
            # dL/dW
            grad_w = (1/sample_size)*np.dot(X.T, prob - Y_onehot)
            grad_b =  (1/sample_size)*np.sum(prob - Y_onehot, axis=0)

            self.weights = self.weights - (self.learning_rate*grad_w)
            self.bias = self.bias - (self.learning_rate*grad_b)

            # Computing cross-entropy loss
            if multinomial:
                loss = self.cross_entropy_loss(prob, Y_onehot)
            else:
                loss = self.binary_cross_entropy_loss(prob, Y_onehot)

            if verbose:
                print(f"Epoch: {i+1} - Loss: {loss}")

        model = {
            "feature_weights": {
                "weights": self.weights,
                "bias": self.bias,
                "Y_to_categorical": self.Y_to_categorical
            },
            "Feature": features_lr_class
        }
        ## Save the model
        self.save_model(model)
        return model


    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_fixle: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """

        feature_weights = model["feature_weights"]
        Feature_LR_class = model["Feature"]

        # Read Input File
        tokenized_text = Feature_LR_class.read_inference_file(input_file)
        # Replace words that are not in vocabulary with OOV (Out-of-Vocabulary)
        # token
        updated_text = []
        for sentence in tokenized_text:
            tmp = Feature_LR_class.replace_unknown_word_with_oov(sentence)
            updated_text.append(tmp)      
        tokenized_text = updated_text
        
        X = []

        # Get features from inference file
        for sentence in tokenized_text:
            # Transform dataset to TF-IDF space
            # Return features with format (1, size_vocabulary)
            X_sentence = Feature_LR_class.get_features(sentence, Feature_LR_class.idf)

            # Concatenate A and B vertically
            X.append(X_sentence)

        X = np.vstack(X)

        # Prediction
        multinomial = True if len(feature_weights['Y_to_categorical'].keys()) > 2 else False
        preds_numerical = self.predict(X, feature_weights['weights'], feature_weights['bias'], multinomial)
        # Map indexes to Categorical space
        preds_label = []
        probs = self.predict_prob(X, feature_weights['weights'], feature_weights['bias'], multinomial)
        for y in preds_numerical:
            tmp = feature_weights['Y_to_categorical'][y]
            preds_label.append(tmp)
        
        return preds_label