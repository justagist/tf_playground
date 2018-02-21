import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

device_string = "/cpu:0"


def load_mnist_data():
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

class AutoEncoder:

    def __init__(self, data):

        self._data = data
        self._define_model()

        self._define_loss()

        self._init_op = tf.global_variables_initializer()
        self._train_network()
        self._plot_training_losses()

    def _define_model(self):

        with tf.device(device_string):
            self._input = tf.placeholder(tf.float32, [None, 784])

        self._create_encoder_network()
        self._create_decoder_network()

    def _define_loss(self):

        with tf.device(device_string):
            self._true_output = tf.placeholder(tf.float32, [None, 784]) # actual answer
            self._pv = tf.placeholder(tf.float32, [1, 2]) # Sparsity prob
            self._beta = tf.placeholder(tf.float32, [1, 1]) # Sparsity penalty (lagrange multiplier)

        # Aditional loss for penalising high activations (http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
        # with tf.device(device_string):
        #     p = tf.nn.softmax(encoder_op)
        #     kl_divergence = tf.reduce_mean(tf.mul(self._pv,tf.log(tf.div(self._pv,p))))
        #     sparsity_loss = tf.mul(self._beta,kl_divergence)

        with tf.device(device_string):
            # add_n gives the sum of tensors
            weight_decay_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            squared_loss = tf.reduce_sum(tf.square(self._decode - self._true_output))

        with tf.device(device_string):
            self._loss_op = tf.reduce_mean(squared_loss) + 0.1*weight_decay_loss #+ sparsity_loss


        with tf.device(device_string):
            self._train_op = tf.train.AdadeltaOptimizer(learning_rate=0.1, rho=0.1, epsilon=0.0001).minimize(self._loss_op)


    def _train_network(self, save_values = True, show_train = False, epochs = 1000):

        self._sess = tf.Session()
        with tf.device(device_string): 
            self._sess.run(self._init_op)

        loss = [0]*epochs
        with tf.device(device_string): 
            for i in range(epochs):
                batch_xs, _ = mnist.train.next_batch(100)
                #_, self._loss[i] = sess.run([train_op, self._loss_op], feed_dict={x: batch_xs, self._true_output: batch_xs})
                _, loss[i] = self._sess.run([self._train_op, self._loss_op], feed_dict={self._input: batch_xs, self._true_output: batch_xs, self._pv: [[0.02,0.98]], self._beta: [[0.1]]})
                
                if i%(int(epochs/10))==0:
                    print "Training: {0:.0f}%".format(i/float(epochs)*100)
                    if show_train:
                        idx = 3
                        out_code, out_decode = self._sess.run([self._encoder_op,self._decode], feed_dict={self._input: np.expand_dims(mnist.test.images[idx,:],0)})
                        plt.subplot(1,4,1)
                        plt.imshow(np.reshape(mnist.test.images[idx,:],(28,28)))
                        plt.subplot(1,4,2)
                        plt.imshow(np.reshape(out_decode,(28,28)))
                        
                        idx = 5
                        out_code, out_decode = self._sess.run([self._encoder_op,self._decode], feed_dict={self._input: np.expand_dims(mnist.test.images[idx,:],0)})
                        plt.subplot(1,4,3)
                        plt.imshow(np.reshape(mnist.test.images[idx,:],(28,28)))
                        plt.subplot(1,4,4)
                        plt.imshow(np.reshape(out_decode,(28,28)))
                        
                        plt.show()
        self._loss = loss

    def _plot_training_losses(self, loss = None):
        if loss is None:
            loss = self._loss
        print "plotting_losses"
        plt.plot(np.r_[loss].ravel())
        plt.show()

    def _create_encoder_network(self):

        # Create an encoder network. Encoder layer: i/p = 784D; o/p = 50D.
        # Bottleneck layer: i/p = 50D; o/p == 2D. 
        # Gives out a 'code' of the input (image)
        with tf.variable_scope('encoder'):
            ## Encoder weights and bias
            W_fc1 = tf.Variable(tf.random_uniform([784,50], dtype=tf.float32))
            b_fc1 = tf.Variable(tf.random_uniform([50], dtype=tf.float32)) 
            
            ## Bottleneck weights and bias
            W_fc2 = tf.Variable(tf.random_uniform([50,2], dtype=tf.float32))
            b_fc2 = tf.Variable(tf.random_uniform([2], dtype=tf.float32)) 
        
        # connecting the layers
        h1_enc = tf.nn.tanh(tf.matmul(self._input, W_fc1) + b_fc1)
        self._encoder_op = tf.nn.tanh(tf.matmul(h1_enc, W_fc2) + b_fc2)

    def _create_decoder_network(self):

        # Create decoder network to decode the code and output the actual image
        with tf.variable_scope('decoder'):

            code_in = tf.placeholder(tf.float32,[None,2])
            
            W_fc1 = tf.Variable(tf.random_uniform([2,50], dtype=tf.float32))
            b_fc1 = tf.Variable(tf.random_uniform([50], dtype=tf.float32)) 
            
            W_fc2 = tf.Variable(tf.random_uniform([50,784], dtype=tf.float32))
            b_fc2 = tf.Variable(tf.random_uniform([784], dtype=tf.float32)) 
            
        h1_dec = tf.nn.tanh(tf.matmul(self._encoder_op, W_fc1) + b_fc1)
        self._decode = tf.nn.tanh(tf.matmul(h1_dec, W_fc2) + b_fc2)

        h1_dec = tf.nn.tanh(tf.matmul(code_in, W_fc1) + b_fc1)
        self._decoder = tf.nn.tanh(tf.matmul(h1_dec, W_fc2) + b_fc2)


if __name__ == '__main__':
    
    mnist = load_mnist_data()

    aen = AutoEncoder(mnist)

