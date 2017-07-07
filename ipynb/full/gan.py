import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from full.misc import DataSaver
import tensorflow as tf

NAME = 'gan'

class GAN:
    # The size of the noise vector
    NOISE_SIZE = 100
    HIDDEN_SIZE = 128
    IMAGE_SIZE = 28*28
    BATCH_SIZE = 128
    N_DIGITS = 10

    def __init__(self):
        imgDir = f'out/{NAME}/imgs'
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
        self.modelName = 'GAN'
        self.dataSaver = DataSaver(NAME)

    def buildGenerator(self):
        # 1st layer's weights and bias
        self.G_W1 = tf.get_variable('G_W1',
            shape=[self.NOISE_SIZE + self.N_DIGITS, self.HIDDEN_SIZE],
            initializer=tf.contrib.layers.xavier_initializer())

        self.G_b1 =  tf.get_variable('G_b1',
            shape=[self.HIDDEN_SIZE],
            initializer=tf.zeros_initializer())

        # 2nd layer's weights and bias
        self.G_W2 = tf.get_variable('G_W2',
            shape=[self.HIDDEN_SIZE, self.IMAGE_SIZE],
            initializer=tf.contrib.layers.xavier_initializer())

        self.G_b2 = tf.get_variable('G_b2',
            shape=[self.IMAGE_SIZE],
            initializer=tf.zeros_initializer())

        # The trainable generator variables
        self.trainable_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

    def generator(self, z, y):
        inputs = tf.concat(axis=1, values=[z, y])

        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob, name='G_sample')
        return G_prob

    def buildDiscriminator(self):
        # 1st layer's weights and bias
        self.D_W1 = tf.get_variable('D_W1',
            shape=[self.IMAGE_SIZE + self.N_DIGITS, self.HIDDEN_SIZE],
            initializer=tf.contrib.layers.xavier_initializer())

        self.D_b1 = tf.get_variable('D_b1',
            shape=[self.HIDDEN_SIZE],
            initializer=tf.zeros_initializer())

        # 2nd layer's weights and bias
        self.D_W2 = tf.get_variable('D_W2',
            shape=[self.HIDDEN_SIZE, 1],
            initializer=tf.contrib.layers.xavier_initializer())

        self.D_b2 = tf.get_variable('D_b2',
            shape=[1],
            initializer=tf.zeros_initializer())

        # The trainable discriminator variables
        self.trainable_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]  

    def discriminator(self, x, y):
        inputs = tf.concat(axis=1, values=[x, y])  

        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def buildModel(self):
        # The input image
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.X = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE], name='X')
        # The input vector of noise
        self.Z = tf.placeholder(tf.float32, shape=[None, self.NOISE_SIZE], name='Z')  
        # One-hot vector of digit type to generate
        self.Y = tf.placeholder(tf.float32, shape=[None, self.N_DIGITS], name='Y')

        self.buildGenerator()
        self.buildDiscriminator()

        # Image created by the generator
        self.G_sample = self.generator(self.Z, self.Y)

        # Descriminator's output for the real MNIST image
        D_real, D_logit_real = self.discriminator(self.X, self.Y)
        # Descriminator's output for the generated MNIST image
        D_fake, D_logit_fake = self.discriminator(self.G_sample, self.Y)

        # Descriminator wants high probability for the real image
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_real,
                labels=tf.ones_like(D_logit_real)))
        # Descriminator also wants low probability for the generated image
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_fake,
                labels=tf.zeros_like(D_logit_fake)))
        # We sum these to get our total descriminator loss
        self.D_loss = D_loss_real + D_loss_fake

        # Generator wants high probability for the generated image
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_fake,
                labels=tf.ones_like(D_logit_fake)))

        # The optimizer for each net
        self.D_optimizer = tf.train.AdamOptimizer().minimize(self.D_loss,
            var_list=self.trainable_D,
            global_step=self.global_step)
        self.G_optimizer = tf.train.AdamOptimizer().minimize(self.G_loss,
            var_list=self.trainable_G)

    @staticmethod
    def plot(samples):
        '''Plots a grid of 16 generated images.
        '''
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig

    @staticmethod
    def sample_Z(m, n):
        '''Returns a uniform sample of values between
        -1 and 1 of size [m, n].
        '''
        return np.random.uniform(-1., 1., size=[m, n])  

    def train(self, writeImages=True):
        from tensorflow.examples.tutorials.mnist import input_data
        # Fetch MNIST Dataset using the supplied Tensorflow Utility Function
        mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)     

        image_y = np.zeros([self.N_DIGITS, self.N_DIGITS])
        for i in range(self.N_DIGITS):
            image_y[i][i] = 1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ## Restore checkpoint and write graph
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(f'out/{NAME}/checkpoints/')
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoring from checkpoint")
                saver.restore(sess, ckpt.model_checkpoint_path)  
                writer = tf.summary.FileWriter(f'out/{NAME}/graphs/{self.modelName}', sess.graph)
            ##

            # Image counter
            it = self.global_step.eval()
            while True:
                # Save out image of 16 generated digits
                
                if writeImages and it % 1000 == 0:
                    samples = sess.run(self.G_sample,
                        feed_dict={
                            self.Z: self.sample_Z(self.N_DIGITS, self.NOISE_SIZE),
                            self.Y: image_y
                        })
                    fig = self.plot(samples)
                    imgNum = str(it // 1000).zfill(3)
                    plt.savefig(f'out/{NAME}/imgs/{imgNum}.png', bbox_inches='tight')
                    plt.close(fig)
                    saver.save(sess, f'out/{NAME}/checkpoints/{self.modelName}')
                    self.dataSaver.save()


                # Get a batch of real MNIST images
                X_batch, Y_batch = mnist.train.next_batch(self.BATCH_SIZE)

                # Run our optimizers
                _, D_loss_curr = sess.run([self.D_optimizer, self.D_loss],
                                          feed_dict={self.X: X_batch,
                                                     self.Z: self.sample_Z(self.BATCH_SIZE, self.NOISE_SIZE),
                                                     self.Y: Y_batch })
                _, G_loss_curr = sess.run([self.G_optimizer, self.G_loss],
                                          feed_dict={self.Z: self.sample_Z(self.BATCH_SIZE, self.NOISE_SIZE),
                                                     self.Y: Y_batch})

                if it % 100 == 0:
                    self.dataSaver.add('Descriminator Loss', float(D_loss_curr), it)
                    self.dataSaver.add('Generator Loss', float(G_loss_curr), it)

                # Report loss
                if it % 1000 == 0:
                    print('Iter: {}'.format(it))
                    print('D loss: {:.4}'. format(D_loss_curr))
                    print('G_loss: {:.4}'.format(G_loss_curr))
                    print()

                it += 1

def graph():
    from matplotlib import pyplot as plt
    plt.title("100 Iteration Loss Average")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    dataSaver = DataSaver(NAME)
    dataSaver.graph(['Descriminator Loss', 'Generator Loss'], AVG_WIN=1)

def generate(modelName='GAN'):
    N_DIGITS = 10
    image_y = np.zeros([N_DIGITS,N_DIGITS])
    for i in range(N_DIGITS):
        image_y[i][i] = 1

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f'out/{NAME}/checkpoints/{modelName}.meta')
        ckpt = tf.train.get_checkpoint_state(f'out/{NAME}/checkpoints/')
        saver.restore(sess, ckpt.model_checkpoint_path)  
        graph = tf.get_default_graph()
        global_step = graph.get_tensor_by_name('global_step:0')
        print("Generating at iter {}".format(global_step.eval()))
        G_sample = graph.get_tensor_by_name('G_sample:0')
        Z = graph.get_tensor_by_name('Z:0')
        Y = graph.get_tensor_by_name('Y:0')
        samples = sess.run(G_sample,
            feed_dict={
                Z: GAN.sample_Z(N_DIGITS, GAN.NOISE_SIZE),
                Y: image_y
            })
        fig = GAN.plot(samples)
        plt.show()

def main(args):
    if 'graph' in args:
        graph()
    elif 'generate' in args:
        generate()
    else:
        gan = GAN()
        gan.buildModel()
        gan.train()

if __name__ == '__main__':
    from sys import argv
    main(argv)