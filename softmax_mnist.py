'''
# Softmax regression/ Softmax layer
    - Usually used as the last layer in an NN for classification.
    - Activation function a_i = exp(z_i)/sum(z_all).
    - The exponentials ensure that all the output activations are positive. And the sum in the denominator ensures that the softmax outputs sum to 1.
    - Can think of softmax as a way of rescaling the z_i, and then squishing them together to form a probability distribution.

    @author: JustaGist
    @package: tf_playground
'''


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys


def load_mnist_data():
    return input_data.read_data_sets("MNIST_data/", one_hot=True)


class SoftmaxPredictor:

    def __init__(self):

        self.n_inputs = 784
        self.n_classes = 10

        self._learning_rate = 0.5

        self._model_initialised = False

        self._initialise_predictor()


    def _initialise_predictor(self):

        # ----- initialise placeholders and variables
        self._inputs = tf.placeholder(tf.float32, [None, self.n_inputs]) # inputs - 784 dimensions (28x28 image pixels as one vector) # None means that dim can be of any length (i.e total number of images in this case)

        # ----- weights and biases
        self._W = tf.Variable(tf.zeros([self.n_inputs, self.n_classes])) # 10 classes:- digits 0 to 9
        self._b = tf.Variable(tf.zeros([self.n_classes]))

        # actual output required, the one hot vectors (10 dimensional) denoting the label for an image 
        self._true_labels = tf.placeholder(tf.float32, [None, self.n_classes]) # 10 classes, n images

        # ----- predictions computed using softmax
        self._evidence = tf.matmul(self._inputs, self._W) + self._b # activation function, gives the evidence supporting the claim for a class or against it

        # ----- Prediction tensor
        self._prediction = tf.nn.softmax(self._evidence)  # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis) # here logits = tf.matmul(x,W) + b, axis = -1 (last dim)

        self._loss_function = self.define_loss(method = 2) # cost function


    def define_loss(self, method = 1):

        
        if method == 1: # (numerically less stable)

            cost_function = tf.reduce_mean(-tf.reduce_sum(self._true_labels * tf.log(self._prediction), reduction_indices=[1])) # cross entropy cost function
            ## ----- tf.reduce_mean computes the mean over all the examples in the batch
            ## ----- tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter

        if method == 2: # combines softmax function and cross-entropy cost

            cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._evidence, labels=self._true_labels))

        return cost_function

    def _check_and_initialise_session(self):

        if not self._model_initialised:
            print "Initialising New TF Session"

            self._sess = tf.Session()
            init = tf.global_variables_initializer()
            self._sess.run(init)
            self._model_initialised = True

        else:
            print "TF session found. Continuing with existing TF Session"


    def train(self, lr = None, optimiser = 'grad_desc', epochs = 1000, save_values = False):

        if save_values:
            saver = tf.train.Saver()

        if lr is None:
            lr = self._learning_rate

        def define_optimiser():

            if optimiser == 'grad_desc':
                return tf.train.GradientDescentOptimizer(lr).minimize(self._loss_function) # Backpropogation algorithm. Learning rate lr = 0.5 by default

        def start_training(optimiser_function):

            for i in range(epochs):

                if i%(int(epochs/10))==0:
                    print "Training: {0:.0f}%".format(i/float(epochs)*100)

                batch_xs, batch_ys = mnist.train.next_batch(100) # batch_ys: one hot vectors
                self._sess.run(optimiser_function, feed_dict={self._inputs: batch_xs, self._true_labels: batch_ys})

        optimiser_function = define_optimiser()
        
        self._check_and_initialise_session()

        start_training(optimiser_function)

        if save_values:
            save_path = saver.save(self._sess, save_file)
            print("Model saved in path: %s" % save_path)

    def test_accuracy(self):

        self._check_and_initialise_session()

        correct_prediction = tf.equal(tf.argmax(self._prediction,1), tf.argmax(self._true_labels,1)) # tf.argmax gives index of highest value in a tensor along an axis

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # [True, False, True, True] would become [1,0,1,1] which would become 0.75.

        # Test accuracy on test data
        print "Accuracy:", (self._sess.run(accuracy, feed_dict={self._inputs: mnist.test.images, self._true_labels: mnist.test.labels}))*100,'%'


    def test_prediction(self, image_idx, visualise_image = True):

        import numpy as np

        self._check_and_initialise_session()
                
        pred = self._sess.run(tf.argmax(self._prediction,1), feed_dict={self._inputs: np.expand_dims(mnist.test.images[image_idx,:],0)})

        print "Prediction:", pred, "Actual:",np.argmax(mnist.test.labels[image_idx,:],-1)

        if visualise_image:
            import matplotlib.pyplot as plt
            plt.imshow(np.reshape(mnist.test.images[image_idx,:],(28,28)))
            plt.show()

    def load_saved_model(self):

        self._check_and_initialise_session()

        saver = tf.train.Saver()
        saver.restore(self._sess, load_file)
        print "Restored Session Model from", load_file

if __name__ == '__main__':

    save_file = "_training_saves/softmax_test_mnist.ckpt"
    load_file = "_training_saves/softmax_default_mnist.ckpt"
    
    if len(sys.argv) < 2:
        print "USAGE: python softmax_mnist.py train [num_epochs] [save?] [save_path]\n\t or: python softmax_mnist.py test [MNIST test datatset img index]"



    else:
        mnist = load_mnist_data()
        softmax = SoftmaxPredictor()
        if sys.argv[1] == 'train':
            epochs = 1000
            save = False
            if len(sys.argv) > 2:
                epochs = int(sys.argv[2]) if (sys.argv[2] != '0') else epochs
                if len(sys.argv) > 3:
                    save = bool(int(sys.argv[3]))
                    if len(sys.argv) > 4 and save == True:
                        save_file = sys.argv[4]
                        save_file = '_training_saves/'+ save_file if '_training_saves/' not in save_file else save_file
                        save_file = save_file + '.ckpt' if '.ckpt' not in save_file else save_file
            print "Starting Training with {0:.0f} epochs".format(epochs)
            softmax.train(epochs=epochs, save_values = save)

        elif sys.argv[1] == 'test':
            idx = 100
            if len(sys.argv) > 2:
                idx = int(sys.argv[2])
                if len(sys.argv) > 3:
                    load_file = sys.argv[3]
                    load_file = '_training_saves/'+load_file if '_training_saves/' not in load_file else load_file
                    load_file = load_file + '.ckpt' if load_file[:-5] != '.ckpt' else load_file
            softmax.load_saved_model()
            softmax.test_prediction(idx)
    # softmax = SoftmaxPredictor()
    # softmax.test_accuracy()
    # softmax.train()
    # softmax.test_accuracy()
    # softmax.test_prediction(389)
