'''
# Multilayer Convolution Network for predicting the number from the mnist dataset.

    @author: JustaGist
    @package: tf_playground

'''

import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def load_mnist_data():
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

class MultiConvNetClassifier:

    def __init__(self):

        self.input_dims = [28,28] # ----- 28x28 image flattened
        self.n_classes = 10 # ----- classes (0 to 9)

        self._initialise_model()

        self._session_initialised = False


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
            # ----- 2d convolution over image x with kernel W using zero padding and stride 1
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            # ----- max pooling over 2x2 blocks
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
        # ----- combines softmax function and cross-entropy cost
        self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv_))

    def _define_accuracy_measure(self):
        correct_prediction = tf.equal(tf.argmax(self.y_conv_,1), tf.argmax(self.y_,1))
        self.accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, epochs = 1000, eval_accuracy = True, plot_losses = True, learning_rate = 1e-4, save_values = True):

        if save_values:
            saver = tf.train.Saver()

        def define_optimiser():
            return tf.train.AdamOptimizer(learning_rate).minimize(self._cross_entropy)

        _train_step = define_optimiser()

        self._check_and_initialise_session()        

        i = 0
        training_done = False

        loss = [0]*epochs
    
        while not training_done:

            try:
                batch = mnist.train.next_batch(50)

                if eval_accuracy and i%100==0:

                    train_accuracy = self._sess.run(self.accuracy_, feed_dict={self.x_:batch[0], self.y_: batch[1], self.keep_prob_: 1.0})
                    print "iter:", i, "Training: {0:.0f}%".format(float(i/float(epochs))*100), "training accuracy %g"%(train_accuracy)

                _, loss[i] = self._sess.run([_train_step, self._cross_entropy], feed_dict={self.x_: batch[0], self.y_: batch[1], self.keep_prob_: 0.5})

                i+=1

                if i >= epochs:
                    training_done = True

            except KeyboardInterrupt:
                break

        self._loss = loss

        if save_values:
            save_path = saver.save(self._sess, save_file)
            print("Model saved in path: %s" % save_path)

        if plot_losses:
            self._plot_training_losses()

        batch = mnist.train.next_batch(50)
        train_accuracy = self._sess.run(self.accuracy_, feed_dict={self.x_:batch[0], self.y_: batch[1], self.keep_prob_: 1.0})
        print("Final training accuracy %g"%(train_accuracy))


    def _plot_training_losses(self):

        print "plotting_losses"
        plt.plot(np.r_[self._loss].ravel())
        plt.show()

    def _check_and_initialise_session(self):

        if not self._session_initialised:
            print "Initialising New TF Session"

            self._sess = tf.Session()
            init = tf.global_variables_initializer()
            self._sess.run(init)
            self._session_initialised = True

        else:
            print "TF session found. Continuing with existing TF Session"

    def load_saved_model(self):

        self._check_and_initialise_session()

        saver = tf.train.Saver()
        saver.restore(self._sess, load_file)
        print "Restored Session Model from", load_file

    def test_prediction(self, image_idx, visualise_image = True):

        self._check_and_initialise_session()
                
        pred = self._sess.run(tf.argmax(self.y_conv_,1), feed_dict={self.x_: np.expand_dims(mnist.test.images[image_idx,:],0), self.keep_prob_: 1.0})

        print "Prediction:", pred, "Actual:",np.argmax(mnist.test.labels[image_idx,:],-1)

        if visualise_image:
            plt.imshow(np.reshape(mnist.test.images[image_idx,:],(28,28)))
            plt.show()


if __name__ == '__main__':

    save_file = "_training_saves/mcn_classifier_test_mnist.ckpt"
    load_file = "_training_saves/mcn_classifier_default_mnist.ckpt"

    if len(sys.argv) < 2:
        print "USAGE: python mcn_classifier_mnist.py train [num_epochs] [save?] [save_path]\n\t or: python mcn_classifier_mnist.py test [MNIST test datatset img index]"

    else:
        mnist = load_mnist_data()
        mcn = MultiConvNetClassifier()
        if sys.argv[1] == 'train':
            epochs = 1000
            save = True
            if len(sys.argv) > 2:
                epochs = int(sys.argv[2]) if (sys.argv[2] != '0') else epochs
                if len(sys.argv) > 3:
                    save = bool(int(sys.argv[3]))
                    if len(sys.argv) > 4 and save == True:
                        save_file = sys.argv[4]
                        save_file = '_training_saves/'+ save_file if '_training_saves/' not in save_file else save_file
                        save_file = save_file + '.ckpt' if '.ckpt' not in save_file else save_file
            print "Starting Training with {0:.0f} epochs".format(epochs)
            mcn.train(epochs=epochs, save_values = save)

        elif sys.argv[1] == 'test':
            idx = 9749 # softmax fails with this but mcn succeeds
            if len(sys.argv) > 2:
                idx = int(sys.argv[2])
                if len(sys.argv) > 3:
                    load_file = sys.argv[3]
                    load_file = '_training_saves/'+load_file if '_training_saves/' not in load_file else load_file
                    load_file = load_file + '.ckpt' if load_file[:-5] != '.ckpt' else load_file
            mcn.load_saved_model()
            mcn.test_prediction(idx)

        else:
            print "Invalid Usage."
            print "USAGE: python mcn_classifier_mnist.py train [num_epochs] [save?] [save_path]\n\t or: python mcn_classifier_mnist.py test [MNIST test datatset img index]"