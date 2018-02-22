'''
# Softmax regression/ Softmax layer
    - Usually used as the last layer in an NN for classification.
    - Activation function a_i = exp(z_i)/sum(z_all).
    - The exponentials ensure that all the output activations are positive. And the sum in the denominator ensures that the softmax outputs sum to 1.
    - Can think of softmax as a way of rescaling the z_i, and then squishing them together to form a probability distribution.

'''


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784]) # inputs - 784 dimensions (28x28 image pixels as one vector) # None means that dim can be of any length (i.e total number of images in this case)

# weights and biases
W = tf.Variable(tf.zeros([784, 10])) # 10 classes:- digits 0 to 9
b = tf.Variable(tf.zeros([10]))

# predictions computed using softmax
evidence = tf.matmul(x, W) + b # activation function, gives the evidence supporting the claim for a class or against it
y = tf.nn.softmax(evidence)  # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis) # here logits = tf.matmul(x,W) + b, axis = -1 (last dim)

# Cost function
# actual output required, the one hot vectors (10 dimensional) denoting the label for an image 
y_ = tf.placeholder(tf.float32, [None, 10]) # 10 classes, n images

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))