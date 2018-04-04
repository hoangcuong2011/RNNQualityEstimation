# Example for my blog post at:
# http://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
import sets
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceLabelling:

    def __init__(self, data, target, dropout, num_hidden=50, num_layers=1):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        network = tf.nn.rnn_cell.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.nn.rnn_cell.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(network, data, dtype=tf.float32)
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        SequenceLabelling.prediction = prediction
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(
            self.target * tf.log(self.prediction), [1, 2])
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)



def get_length_corpus(filename):
    with open(filename) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        #print(len(content)) 
    return len(content)

def get_maximum_length(filename):
    
    with open(filename) as f:
        maxlength_sentence = 0
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]        
        for i in range(0, len(content)):
            k = len(content[i].split(" "))
            if(k>maxlength_sentence):
                #print(content[i])
                #print(maxlength_sentence)
                maxlength_sentence = k
    return maxlength_sentence


def create_dataset(filename, corpus_file, maxlength_sentence, number_of_features):
    dataset = np.loadtxt(corpus_file, delimiter=",")
    x_valid = dataset[:,0:number_of_features]
    y_valid = dataset[:,614].reshape(-1,1)
    count = 0
    number_of_features = number_of_features
    length_sequences = []

    with open(filename) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        print(len(content))
        input_corpus = np.zeros((len(content), maxlength_sentence, number_of_features))
        output_corpus = np.zeros((len(content), maxlength_sentence, 2))

        for i in range(0, len(content)):
            k = len(content[i].split(" "))
            length_sequences.append(k)
            for l in range(0, k):
                input_corpus[i,l] = x_valid[count]
                output = np.zeros(2)
                if y_valid[count]==1:
                    output[1] = 1
                else:
                    output[0] = 1
                output_corpus[i,l] = output
                count = count+1

    dev_input = input_corpus
    dev_output = output_corpus

    print(dev_input.shape)
    #print(dev_input)

    print(type(dev_input))
    print(type(dev_output))

    return dev_input, dev_output, length_sequences
    



def compute_scores(flat_true, flat_pred):
    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    print("F1-BAD: ", f1_bad, "F1-OK: ", f1_good)
    print("F1-score multiplied: ", f1_bad * f1_good)


def convertContinuoustoOutput(y_target, y_preds, test_length_sequences):
    y_preds_binary = []
    y_target_binary = []

    for i in range(0, len(y_preds)):
        for j in range(0, test_length_sequences[i]):
            if(y_preds[i][j][0]>0.5):
                x = 0
            else:
                x = 1
            y_preds_binary.append(x)

            if(y_target[i][j][0]>0.5):
                x = 0
            else:
                x = 1
            y_target_binary.append(x)

    #print(y_preds_binary)    
    #print(y_target_binary)    
    return y_preds_binary, y_target_binary



def standardize_data(X_train, X_test, X_valid):
    unique_X_train = np.unique(X_train, axis=0)
    X_mean = np.mean(unique_X_train, axis=0)
    #print(X_mean)
    X_std = np.std(unique_X_train, axis=0)+0.0000001 #a small noise
    #print(X_std)
    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid
def rewriteFile(filename, x_1, x_2, x_3, y):
    file = open(filename,"w")
    for i in range(0, len(x_1)):
        for j in range(0, len(x_1[i])):
            file.write(str(x_1[i][j]))
            file.write(",")

        for j in range(0, len(x_2[i])):
            file.write(str(x_2[i][j]))
            file.write(",")

        for j in range(0, len(x_3[i])):
            file.write(str(x_3[i][j]))
            file.write(",")
        file.write(str(y[i][0]))
        file.write("\n")
    file.close()

def standardize_data_and_rewrite():
    dataset = np.loadtxt("test", delimiter=",")
    x_1_test = np.concatenate([dataset[:,0:3], dataset[:,603:614]], axis=1)
    x_2_test = dataset[:,3:303]
    x_3_test = dataset[:,303:603]    
    y_test = dataset[:,614].reshape(-1,1)



    dataset = np.loadtxt("dev", delimiter=",")
    y_valid = dataset[:,614].reshape(-1,1)

    x_1_valid = np.concatenate([dataset[:,0:3], dataset[:,603:614]], axis=1)
    x_2_valid = dataset[:,3:303]
    x_3_valid = dataset[:,303:603]

    dataset = np.loadtxt("train", delimiter=",")
    x_1_train = np.concatenate([dataset[:,0:3], dataset[:,603:614]], axis=1)

    x_2_train = dataset[:,3:303]
    x_3_train = dataset[:,303:603]

    y_train = dataset[:,614].reshape(-1,1)

    
    x_1_train, x_1_test, x_1_valid = standardize_data(x_1_train, x_1_test, x_1_valid)

    rewriteFile("test.rewrite", x_1_test, x_2_test, x_3_test, y_test)
    rewriteFile("train.rewrite", x_1_train, x_2_train, x_3_train, y_train)
    rewriteFile("dev.rewrite", x_1_valid, x_2_valid, x_3_valid, y_valid)


if __name__ == '__main__':
    #standardize_data_and_rewrite()
    train_data, train_target, train_length_sequences = create_dataset("train.tags", "train.rewrite", get_maximum_length("all.tags"), 614)
    dev_data, dev_target, dev_length_sequences = create_dataset("dev.tags", "dev.rewrite", get_maximum_length("all.tags"), 614)
    test_data, test_target, test_length_sequences = create_dataset("test.tags", "test.rewrite", get_maximum_length("all.tags"), 614)
    
    #train_data, dev_data, test_data = standardize_data(train_data, dev_data, test_data)

    print("Finish")

    _, length, image_size = train_data.shape
    print(length)
    print(image_size)
    minibatch_size = 250
    num_classes = train_target.shape[2]
    data = tf.placeholder(tf.float32, [None, length, image_size])
    target = tf.placeholder(tf.float32, [None, length, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceLabelling(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for epoch in range(50):
        shuffle = np.arange(len(train_data))    
        #print(shuffle)
        np.random.shuffle(shuffle)
        #print(shuffle)
        x_train_shuffle = train_data[shuffle]
        y_train_shuffle = train_target[shuffle]
        data_indx = 0
        while data_indx<len(train_target):
            lastIndex = data_indx + minibatch_size
            if lastIndex>=len(train_target):
                lastIndex = len(train_target)
            indx_array = np.mod(np.arange(data_indx, lastIndex), x_train_shuffle.shape[0])
            data_indx += minibatch_size

            sess.run(model.optimize, {
                data: x_train_shuffle[indx_array], target: y_train_shuffle[indx_array], dropout: 0.5})

        prediction, error = sess.run([model.prediction, model.error], {data: dev_data, target: dev_target, dropout: 1})
        y_preds_binary, y_target_binary = convertContinuoustoOutput(dev_target, prediction, dev_length_sequences)
        compute_scores(y_target_binary, y_preds_binary)

        prediction, error = sess.run([model.prediction, model.error], {data: test_data, target: test_target, dropout: 1})
        y_preds_binary, y_target_binary = convertContinuoustoOutput(test_target, prediction, test_length_sequences)
        compute_scores(y_target_binary, y_preds_binary)

        



