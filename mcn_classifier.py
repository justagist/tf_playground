from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# sess = tf.InteractiveSession()

# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])


# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)

# def bias_variable(shape):
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)


# def conv2d(x, W):
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')


# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])

# x_image = tf.reshape(x, [-1,28,28,1])

# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])


# sess.run(tf.global_variables_initializer())

# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.global_variables_initializer())
# for i in range(20000):
#   batch = mnist.train.next_batch(50)
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def load_mnist_data():
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

class MultiConvNetClassifier:

    def __init__(self):

        self.input_dims = [28,28] # ----- 28x28 image flattened
        self.n_classes = 10 # ----- classes (0 to 9)

        self._sess = tf.Session()

        self._initialise_model()


    def _initialise_model(self):

        self.x_ = tf.placeholder(tf.float32, shape=[None, self.input_dims[0]*self.input_dims[1]]) # ----- input (28x28 flattened)
        # ----- reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels
        self.inputs_ = tf.reshape(self.x_, [-1,self.input_dims[0],self.input_dims[1],1])

        self.y_ = tf.placeholder(tf.float32, shape=[None, self.n_classes]) # ----- true output vector (one hot)

        self._create_network()

        self._define_loss_function_and_optimiser()

        self._define_accuracy_measure()
        
    def _create_network(self):

        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)


        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        # ===== Convolution Layer 1: will compute 32 features for each 5x5 patch of the image. 
        W_conv1 = weight_variable([5, 5, 1, 32]) # ----- weights tensor. 1 indicates the input channel (1 for Grayscale, 3 for RGB etc.)
        b_conv1_ = bias_variable([32]) # ----- bias vector

        # ----- convolve input with the weight tensor, add the bias, apply the ReLU function, and take the max pool
        h_conv1_ = tf.nn.relu(conv2d(self.inputs_, W_conv1) + b_conv1_)
        h_pool1_ = max_pool_2x2(h_conv1_) # max_pool done over 2x2 window


        # ===== Convolution Layer 2: will compute 64 features for each 5x5 patch of the image. 
        W_conv2_ = weight_variable([5, 5, 32, 64]) # 32 input channels
        b_conv2_ = bias_variable([64])

        h_conv2_ = tf.nn.relu(conv2d(h_pool1_, W_conv2_) + b_conv2_)
        h_pool2_ = max_pool_2x2(h_conv2_)

        # ===== Densely Connected Layer: a fully-connected layer with 1024 neurons to allow processing on the entire image.
        W_fc1_ = weight_variable([7 * 7 * 64, 1024])
        b_fc1_ = bias_variable([1024])

        # ----- The 7x7 image is reshaped into a batch of vectors, and multiplied with the weight and bias added. And ReLU done.
        h_pool2_flat_ = tf.reshape(h_pool2_, [-1, 7*7*64])
        h_fc1_ = tf.nn.relu(tf.matmul(h_pool2_flat_, W_fc1_) + b_fc1_)

        # ===== Implementing dropout before the readout layer.
        self.keep_prob_ = tf.placeholder(tf.float32) # ----- placeholder for the probability that a neuron's output is not dropped during training. For turning on and off the dropout (for training and testing)
        h_fc1_drop_ = tf.nn.dropout(h_fc1_, self.keep_prob_) # ----- implements dropout according to keep_prob

        # ===== Readout Layer: Output layer for determining the class of the input
        W_fc2_ = weight_variable([1024, 10])
        b_fc2_ = bias_variable([10])

        self.y_conv_ = tf.matmul(h_fc1_drop_, W_fc2_) + b_fc2_ # ----- predicted output

    def _define_loss_function_and_optimiser(self):

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv_)) # ----- combines softmax function and cross-entropy cost

        self._train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # learning rate 1e-4

    def _define_accuracy_measure(self):

        correct_prediction = tf.equal(tf.argmax(self.y_conv_,1), tf.argmax(self.y_,1))
        self.accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train_network(self, epochs = 1000, eval_accuracy = True):
        init = tf.global_variables_initializer()
        self._sess.run(init)
        for i in range(epochs): 

            batch = mnist.train.next_batch(50)

            if eval_accuracy:
                if i%100 == 0:

                    train_accuracy = self.accuracy_.eval(feed_dict={self.x_:batch[0], self.y_: batch[1], self.keep_prob_: 1.0})

                    print("step %d, training accuracy %g"%(i, train_accuracy))

            self._train_step.run(feed_dict={self.x_: batch[0], self.y_: batch[1], self.keep_prob: 0.5})


if __name__ == '__main__':
    
    mnist = load_mnist_data()
    mcn = MultiConvNetClassifier()

    mcn.train_network()