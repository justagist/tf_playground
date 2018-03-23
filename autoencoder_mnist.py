'''
    # autoencoder class to learn and predict mnist images: Reduces 28x28 image to a 2D value and learns to predict the class from it.

    @author: JustaGist
    @package: tf_playground
'''
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_data():
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

class AutoEncoder:

    def __init__(self):

        self._data = mnist
        self._define_model()
        self._sess = tf.Session()
        # self._saver = tf.train.Saver()
        self._use_saved_model = False
        # self.plot_training_losses()


    def _define_model(self):

        self._input = tf.placeholder(tf.float32, [None, 784])

        self._create_encoder_network()
        self._create_decoder_network()
        self._define_loss_functions()

    def _define_loss_functions(self):

        self._true_output = tf.placeholder(tf.float32, [None, 784]) # actual answer
        self._pv = tf.placeholder(tf.float32, [1, 2]) # Sparsity prob
        self._beta = tf.placeholder(tf.float32, [1, 1]) # Sparsity penalty (lagrange multiplier)

        # Aditional loss for penalising high activations (http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
        # p = tf.nn.softmax(encoder_op)
        # kl_divergence = tf.reduce_mean(tf.mul(self._pv,tf.log(tf.div(self._pv,p))))
        # sparsity_loss = tf.mul(self._beta,kl_divergence)

        # add_n gives the sum of tensors
        weight_decay_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        squared_loss = tf.reduce_sum(tf.square(self._train_decode - self._true_output))

        self._loss_op = tf.reduce_mean(squared_loss) + 0.1*weight_decay_loss #+ sparsity_loss


        self._train_op = tf.train.AdadeltaOptimizer(learning_rate=0.1, rho=0.1, epsilon=0.0001).minimize(self._loss_op)


    def train_network(self, save_values = True, show_train = False, epochs = 10000):

        if save_values:
            saver = tf.train.Saver()

        self._init_op = tf.global_variables_initializer()
        self._sess.run(self._init_op)

        loss = [0]*epochs
        # with tf.device(device_string): 
        i = 0
        training_done = False
        while not training_done:
            try:
                batch_xs, _ = mnist.train.next_batch(100)
                #_, self._loss[i] = sess.run([train_op, self._loss_op], feed_dict={x: batch_xs, self._true_output: batch_xs})
                _, loss[i] = self._sess.run([self._train_op, self._loss_op], feed_dict={self._input: batch_xs, self._true_output: batch_xs, self._pv: [[0.02,0.98]], self._beta: [[0.1]]})
                
                i+=1
                if i%(int(epochs/10))==0:
                    print "Training: {0:.0f}%".format(i/float(epochs)*100)
                    if show_train:
                        idx = 3
                        out_code, out_decode = self._sess.run([self._encoder_op,self._train_decode], feed_dict={self._input: np.expand_dims(mnist.test.images[idx,:],0)})
                        plt.subplot(1,4,1)
                        plt.imshow(np.reshape(mnist.test.images[idx,:],(28,28)))
                        plt.subplot(1,4,2)
                        plt.imshow(np.reshape(out_decode,(28,28)))
                        
                        idx = 5
                        out_code, out_decode = self._sess.run([self._encoder_op,self._train_decode], feed_dict={self._input: np.expand_dims(mnist.test.images[idx,:],0)})
                        plt.subplot(1,4,3)
                        plt.imshow(np.reshape(mnist.test.images[idx,:],(28,28)))
                        plt.subplot(1,4,4)
                        plt.imshow(np.reshape(out_decode,(28,28)))
                        
                        plt.show()
                if i >= epochs:
                    training_done = True
            except KeyboardInterrupt:
                break

        if save_values:
            save_path = saver.save(self._sess, save_file)
            print("Model saved in path: %s" % save_path)

        self._loss = loss

    def plot_training_losses(self, loss = None):
        if loss is None:
            loss = self._loss
        print "plotting_losses"
        plt.plot(np.r_[loss].ravel())
        plt.show()

    def decode(self, code_in, plot=True):
        out_decode = self._sess.run([self._decoder], feed_dict={self._code_in: code_in[0]})[0]

        if plot:
            # plt.subplot(1,2,1)
            plt.imshow(np.reshape(out_decode,(28,28)))
            plt.show()

        return out_decode

    def encode_mnist_image(self, image_idx):

        code = self.encode_image(mnist.test.images[image_idx,:])
        return code

    def encode_image(self, image):

        code = self._sess.run([self._encoder_op], feed_dict={self._input: np.expand_dims(image,0)})
        return code

    def encode_decode_mnist(self, image_idx):

        out_code, out_decode = self._sess.run([self._encoder_op, self._train_decode], feed_dict={self._input: np.expand_dims(mnist.test.images[image_idx,:],0)})
        return out_code, out_decode


    def decode_and_compare(self, code, mnist_image_idx):

        plt.subplot(1,2,1)
        plt.imshow(np.reshape(mnist.test.images[mnist_image_idx,:],(28,28)))

        out_decode = self.decode(code, plot=False)
        plt.subplot(1,2,2)
        plt.imshow(np.reshape(out_decode,(28,28)))
        plt.show()

    def test_autoencoding(self, image_idx):

        self.decode_and_compare(self.encode_mnist_image(image_idx),image_idx)


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

            self._code_in = tf.placeholder(tf.float32,[None,2])
            
            W_fc1 = tf.Variable(tf.random_uniform([2,50], dtype=tf.float32))
            b_fc1 = tf.Variable(tf.random_uniform([50], dtype=tf.float32)) 
            
            W_fc2 = tf.Variable(tf.random_uniform([50,784], dtype=tf.float32))
            b_fc2 = tf.Variable(tf.random_uniform([784], dtype=tf.float32)) 
            

        h1_dec = tf.nn.tanh(tf.matmul(self._encoder_op, W_fc1) + b_fc1)
        self._train_decode = tf.nn.tanh(tf.matmul(h1_dec, W_fc2) + b_fc2) # output decoder while training

        h1_dec = tf.nn.tanh(tf.matmul(self._code_in, W_fc1) + b_fc1)
        self._decoder = tf.nn.tanh(tf.matmul(h1_dec, W_fc2) + b_fc2) # output decoder for testing (requires _code_in placeholder)

    def load_saved_model(self):

        saver = tf.train.Saver()
        saver.restore(self._sess, load_file)
        print "Restored Session Model from", load_file
        self._use_saved_model = True


if __name__ == '__main__':

    save_file = "_training_saves/autoencoder_test_mnist.ckpt"
    load_file = "_training_saves/autoencoder_default_mnist.ckpt"
    
    if len(sys.argv) < 2:
        print "USAGE: python autoencoder_mnist.py train [num_epochs] [save?] [save_path]\n\t or: python autoencoder_mnist.py test [MNIST test datatset img index]"



    else:
        mnist = load_mnist_data()
        aen = AutoEncoder()
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
            aen.train_network(epochs=epochs, save_values = save)

        elif sys.argv[1] == 'test':
            idx = 100
            if len(sys.argv) > 2:
                idx = int(sys.argv[2])
                if len(sys.argv) > 3:
                    load_file = sys.argv[3]
                    load_file = '_training_saves/'+load_file if '_training_saves/' not in load_file else load_file
                    load_file = load_file + '.ckpt' if load_file[:-5] != '.ckpt' else load_file
            aen.load_saved_model()
            aen.test_autoencoding(idx)


    # aen.test_mnist(212)

