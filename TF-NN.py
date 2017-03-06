# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial


import tensorflow as tf
import pandas as pd
import numpy as np

# define dataset
class DataSet(object):
    def __init__(self, inputs, labels, one_hot=False):
        """Construct a DataSet"""
        assert inputs.shape[0] == labels.shape[0], (
            'inputs.shape: %s labels.shape: %s' % (inputs.shape,
                                                   labels.shape))
        self._num_examples = inputs.shape[0]
        self._inputs = inputs
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._inputs = self._inputs[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._inputs[start:end], self._labels[start:end]

#define NN
def add_layer(inputs, in_size, out_size, wname, bname, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name=wname)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name=bname)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return (Weights, biases, outputs)

'''
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
'''

def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()

    ###read data
    data_path = "F:/博士学习/数据/encodingdata/"

    traininputs = pd.read_csv(data_path+"traininput3.csv", header=None, delimiter=",") #traininput dim 10000 * 10
    # normalize train input using scripts
    trainlabels = pd.read_csv(data_path+"trainlabels3.csv", header=None, delimiter=",") #trainlabels dim 10000 * 7
    testinputs = pd.read_csv(data_path+"testinput3.csv", header=None, delimiter=",") #testinput dim 5000 * 10
    # normalize test input using scripts
    testlabels = pd.read_csv(data_path+"testlabels3.csv", header=None, delimiter=",") #testlabels dim 5000 * 7

    data_sets.train = DataSet(traininputs, trainlabels)
    data_sets.test = DataSet(testinputs, testlabels)

    return data_sets

data_sets = read_data_sets()
sess = tf.InteractiveSession()

label_number = 7
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 10])
ys = tf.placeholder(tf.float32, [None, label_number])
# add  first hidden layer
(w1, b1, h1) = add_layer(xs, 10, 2, "w1", "b1", activation_function=tf.nn.relu)
# add  second hidden layer
(w2, b2, h2) = add_layer(h1, 2, 1, "w2", "b2", activation_function=tf.nn.relu)
# add output layer
(w3, b3, prediction) = add_layer(h2, 1, label_number, "w3", "b3", activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.initialize_all_variables().run()

for i in range(300):    # 100 * 100 = 10000 training set
    batch_xs, batch_ys = data_sets.train.next_batch(100)
    train_step.run({xs: batch_xs, ys: batch_ys})
y_pre = sess.run(prediction, feed_dict={xs: data_sets.test.inputs})
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# if i % 5 == 0:
print(accuracy.eval({xs: data_sets.test.inputs, ys: data_sets.test.labels}))

print(w1.eval(sess))
print(b1.eval(sess))

print(w2.eval(sess))
print(b2.eval(sess))

print(w3.eval(sess))
print(b3.eval(sess))
#np.savetxt("w1.csv",w1.eval(sess), delimiter=",")
#np.savetxt("b1.csv",b1.eval(sess), delimiter=",")
#np.savetxt("w2.csv",w2.eval(sess), delimiter=",")
#np.savetxt("b2.csv",b2.eval(sess), delimiter=",")