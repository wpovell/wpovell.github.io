import numpy as np
import tensorflow as tf
import scipy.io
from PIL import Image, ImageOps
import os
import time

NAME = 'style'

def vgg_weights(vgg_layers, layer):
    """ Return the weights and biases already trained by VGG.
    This is all related to the format of the VGG model and isn't that interesting.
    """
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    return W, b.reshape(b.size)

def conv2d_relu(vgg_layers, prev_layer, layer):
    """ Return the Conv2D layer with RELU using the weights, biases from the VGG
    model at 'layer'.
    Inputs:
        vgg_layers: holding all the layers of VGGNet
        prev_layer: the output tensor from the previous layer
        layer: the index to current layer in vgg_layers

    Output:
        relu applied on the convolution.
    """
    # Get the weights from the vgg model for the current layer
    W, b = vgg_weights(vgg_layers, layer)

    # These are consts because they're already trained, we won't be changing them
    W = tf.constant(W, name='weights')
    b = tf.constant(b, name='bias')
    
    conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2d + b)

def avgpool(prev_layer):
    """ Return the average pooling layer. The paper suggests that average pooling
    actually works better than max pooling.
    Input:
        prev_layer: the output tensor from the previous layer

    Output:
        the output of the tf.nn.avg_pool() function.
    """
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                          padding='SAME', name='avg_pool_')

def get_resized_image(img_path, height, width):
    image = Image.open(img_path)
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)

def generate_noise_image(content_image, height, width, noise_ratio=0.6):
    """Take our content image and fuzz it a bit 
    """
    noise_image = np.random.uniform(-20, 20, 
                                    (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F
    """
    F = tf.reshape(F, (M, N))
    return tf.matmul(tf.transpose(F), F)

def single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image
    Output:
        the style loss at a certain layer (which is E_l in the paper)
    """
    N = a.shape[3] # number of filters
    M = a.shape[1] * a.shape[2] # height times width of the feature map
    A = gram_matrix(a, N, M)
    G = gram_matrix(g, N, M)
    return tf.reduce_sum((G - A) ** 2 / ((2 * N * M) ** 2))

def main(CONTENT_IMAGE, STYLE_IMAGE):
    # Output image size
    # Larger images will take longer to train &
    # may not work as well as VGG was trained on small imgs
    IMAGE_HEIGHT = 500
    IMAGE_WIDTH = 666

    # Use variable instead of placeholder because we're training the intial image to make it
    # look like both the content image and the style image
    input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)

    # Load in weights from the pre-trained vgg model
    vgg = scipy.io.loadmat('data/vgg19.mat')
    vgg_layers = vgg['layers']

    # Build up our graph, passing the input_image through our trained weights
    graph = {} 
    graph['conv1_1']  = conv2d_relu(vgg_layers, input_image, 0)
    graph['conv1_2']  = conv2d_relu(vgg_layers, graph['conv1_1'], 2)
    graph['avgpool1'] = avgpool(graph['conv1_2'])
    graph['conv2_1']  = conv2d_relu(vgg_layers, graph['avgpool1'], 5)
    graph['conv2_2']  = conv2d_relu(vgg_layers, graph['conv2_1'], 7)
    graph['avgpool2'] = avgpool(graph['conv2_2'])
    graph['conv3_1']  = conv2d_relu(vgg_layers, graph['avgpool2'], 10)
    graph['conv3_2']  = conv2d_relu(vgg_layers, graph['conv3_1'], 12)
    graph['conv3_3']  = conv2d_relu(vgg_layers, graph['conv3_2'], 14)
    graph['conv3_4']  = conv2d_relu(vgg_layers, graph['conv3_3'], 16)
    graph['avgpool3'] = avgpool(graph['conv3_4'])
    graph['conv4_1']  = conv2d_relu(vgg_layers, graph['avgpool3'], 19)
    graph['conv4_2']  = conv2d_relu(vgg_layers, graph['conv4_1'], 21)
    graph['conv4_3']  = conv2d_relu(vgg_layers, graph['conv4_2'], 23)
    graph['conv4_4']  = conv2d_relu(vgg_layers, graph['conv4_3'], 25)
    graph['avgpool4'] = avgpool(graph['conv4_4'])
    graph['conv5_1']  = conv2d_relu(vgg_layers, graph['avgpool4'], 28)
    graph['conv5_2']  = conv2d_relu(vgg_layers, graph['conv5_1'], 30)
    graph['conv5_3']  = conv2d_relu(vgg_layers, graph['conv5_2'], 32)
    graph['conv5_4']  = conv2d_relu(vgg_layers, graph['conv5_3'], 34)
    graph['avgpool5'] = avgpool(graph['conv5_4'])

    MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    """ MEAN_PIXELS is defined according to description on their github:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8
    'In the paper, the model is denoted as the configuration D trained with scale jittering. 
    The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
    Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
    """

    content_image = get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    content_image = content_image - MEAN_PIXELS

    style_image = get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image = style_image - MEAN_PIXELS

    # percentage weight of the noise for intermixing with the content image
    NOISE_RATIO = 0.6
    initial_image = generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)    

    # Layers used for style features.
    STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    W = [0.5, 1.0, 1.5, 3.0, 4.0] # give more weights to deeper layers.

    with tf.Session() as sess:
        sess.run(input_image.assign(style_image))
        A = sess.run([graph[layer_name] for layer_name in STYLE_LAYERS])

    # Get layer losses
    E = [single_style_loss(A[i], graph[STYLE_LAYERS[i]]) for i in range(len(STYLE_LAYERS))]
    # Linearly combine layer losses
    n_layers = len(STYLE_LAYERS)
    graph['style_loss'] = sum([W[i] * E[i] for i in range(n_layers)])


    # Layer used for content features.
    CONTENT_LAYER = 'conv4_2'

    with tf.Session() as sess:
        sess.run(input_image.assign(content_image)) # assign content image to the input variable
        p = sess.run(graph[CONTENT_LAYER])
    f = graph[CONTENT_LAYER]
    graph['content_loss'] = tf.reduce_sum((f - p) ** 2) / (4.0 * p.size)

    CONTENT_WEIGHT = 0.01
    STYLE_WEIGHT = 1
    graph['total_loss'] = CONTENT_WEIGHT * graph['content_loss'] + STYLE_WEIGHT * graph['style_loss']    

    graph['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    LEARNING_RATE = 2.0
    graph['optimizer'] = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
                            graph['total_loss'], 
                            global_step=graph['global_step'])

    ITERS = 300
    skip_step = 1
    with tf.Session() as sess:
        # Initialize vars
        sess.run(tf.global_variables_initializer())
        # Assign our initial image that we'll be modifying
        sess.run(input_image.assign(initial_image))

        start_time = time.time()
        for index in range(ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20

            sess.run(graph['optimizer'])
            if (index + 1) % skip_step == 0:
                gen_image, total_loss = sess.run([input_image, graph['total_loss']])
                # Readd the mean we subtracted earlier
                gen_image = gen_image + MEAN_PIXELS
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = f'out/{NAME}/{index}.png'
                
                image = gen_image[0] # the image
                image = np.clip(image, 0, 255).astype('uint8')
                scipy.misc.imsave(filename, image)

if __name__ == '__main__':
    from sys import argv
    content = argv[1]
    style = argv[2]
    main(content, style)