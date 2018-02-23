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

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # tf.reduce_mean computes the mean over all the examples in the batch # tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter

# NOTE: tf.nn.softmax_cross_entropy_with_logits_v2(evidence) can be used to combine the above softmax function and cross-entropy cost. This is more stable numerically

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # Backpropogation algorithm. Learning rate 0.5

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train network
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) # batch_ys: one hot vectors
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

'''
Each step of the loop, we get a "batch" of one hundred random data points from our training set. We run train_step feeding in the batches data to replace the placeholders.

Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. Ideally, we'd like to use all our data for every step of training because that would give us a better sense of what we should be doing, but that's expensive. So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.
'''

# Evaluate Model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # tf.argmax gives index of highest value in a tensor along an axis

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # [True, False, True, True] would become [1,0,1,1] which would become 0.75.

# Test accuracy on test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
