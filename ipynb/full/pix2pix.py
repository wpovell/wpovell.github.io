import tensorflow as tf
import numpy as np
import glob
import math
import time
import os

def loadData(path, shuffle=True):
    ## Load data ##
    input_paths = glob.glob(os.path.join(path, '*.jpg')) # All jpgs
    path_queue = tf.train.string_input_producer(input_paths, shuffle=shuffle) # Produces image paths
    reader = tf.WholeFileReader()
    paths, contents = reader.read(path_queue)
    rawInput = tf.image.decode_jpeg(contents)
    rawInput = tf.image.convert_image_dtype(rawInput, dtype=tf.float32)

    # [height, width, channel]
    rawInput.set_shape([None, None, 3])
    width = tf.shape(rawInput)[1]

    def process(r):
        r = tf.image.resize_images(r, [256, 256], method=tf.image.ResizeMethod.AREA)
        return r * 2 - 1# [0, 1] => [-1, 1]

    targets = process(rawInput[:,:width//2,:]) # Left side
    inputs = process(rawInput[:,width//2:,:]) # Right side

    # Batches
    BATCH_SIZE = 1
    paths, inputs, targets = tf.train.batch([paths, inputs, targets], batch_size=BATCH_SIZE)
    steps_per_epoch = int(math.ceil(len(input_paths) / BATCH_SIZE))
    return paths, inputs, targets, steps_per_epoch

paths, inputs, targets, steps_per_epoch = loadData('data/pix2pix/facades/train')
tst_paths, tst_inputs, tst_targets, tst_steps_per_epoch = loadData('data/pix2pix/facades/val')

def conv(batch_input, out_channels, stride):
    ''' Convolve input with given stride. '''
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        # The trainable filter we create for the conv
        conv_filter = tf.get_variable("filter",
                                      [4, 4, in_channels, out_channels],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(0, 0.02))

        padded_input = tf.pad(batch_input,
                              [[0, 0], [1, 1], [1, 1], [0, 0]],
                              mode="CONSTANT")
        # Output of the conv
        conv = tf.nn.conv2d(padded_input,
                            conv_filter,
                            [1, stride, stride, 1],
                            padding="VALID")
        return conv

def deconv(batch_input, out_channels):
    ''' Transposed Convolution. '''
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]

        # The trainable filter we create for the deconv
        conv_filter = tf.get_variable("filter",
                                      [4, 4, out_channels, in_channels],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(0, 0.02))
        # Output of the deconv
        conv = tf.nn.conv2d_transpose(batch_input,
                                      conv_filter,
                                      [batch, in_height * 2, in_width * 2, out_channels],
                                      [1, 2, 2, 1],
                                      padding="SAME")
        return conv

def lrelu(x, a):
    ''' Leaky ReLU.
        x is our tensor.
        a is the magnitude of negative slope for x < 0.
    '''
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(inp):
    ''' Batch Normalization. '''
    with tf.variable_scope("batchnorm"):
        channels = inp.get_shape()[3]
        offset = tf.get_variable("offset",
                                 [channels],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale",
                                [channels],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))

        mean, variance = tf.nn.moments(inp, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(inp, mean, variance,
                                               offset, scale, variance_epsilon=variance_epsilon)
        return normalized

# Number of generator filters
NGF = 64

def create_generator(generator_inputs):
    ''' Creates our generator for the given inputs. '''
    layers = []

    # encoder_1: [batch, 256, 256, 3] => [batch, 128, 128, ngf]
    # This layer doesn't get batchnorm (from paper)
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, NGF, stride=2)
        layers.append(output)

    layer_specs = [
        NGF * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        NGF * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        NGF * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        NGF * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        NGF * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        NGF * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        NGF * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (NGF * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8]       => [batch, 2, 2, ngf * 8 * 2]
        (NGF * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2]   => [batch, 4, 4, ngf * 8 * 2]
        (NGF * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2]   => [batch, 8, 8, ngf * 8 * 2]
        (NGF * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2]   => [batch, 16, 16, ngf * 8 * 2]
        (NGF * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (NGF * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (NGF, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        # Conv layer to connect to
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # First decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer.
                inp = layers[-1]
            else:
                inp = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(inp)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, 3]
    with tf.variable_scope("decoder_1"):
        inp = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(inp)
        output = deconv(rectified, 3)
        output = tf.tanh(output) # Limits output to (-1, 1)
        layers.append(output)

    return layers[-1]

# Number of discriminator filters
NDF = 64

def create_discriminator(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inp = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(inp, NDF, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = NDF * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved) # Limits output to (0, 1), a probability
        layers.append(output)

    return layers[-1]

# Our generated images
with tf.variable_scope("generator"):
    outputs = create_generator(inputs)
with tf.variable_scope("generator", reuse=True):
    tst_outputs = create_generator(tst_inputs)

# Discriminator for our "real" images
with tf.variable_scope("discriminator"):
    predict_real = create_discriminator(inputs, targets)

# Discriminator for our generated images
with tf.variable_scope("discriminator", reuse=True):
    predict_fake = create_discriminator(inputs, outputs)

# To prevent log(0)
EPS = 1e-12

# Discriminator Loss
discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

# Parameters controling how much we weight each loss
L1_WEIGHT = 100.
GAN_WEIGHT = 1.

# Adversarial Loss
gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
# L1 Loss
gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
# Overall Loss
gen_loss = gen_loss_GAN * GAN_WEIGHT + gen_loss_L1 * L1_WEIGHT


# Learning Rate
LR = 0.0002
# Momentum Term
BETA1 = 0.5

# Discriminator training variables
discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
discrim_optim = tf.train.AdamOptimizer(LR, BETA1)
discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

# Makes it so discrim must exec first
with tf.control_dependencies([discrim_train]):
    # All trainable generator variables
    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    gen_optim = tf.train.AdamOptimizer(LR, BETA1)
    gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
    gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

# Maintain moving averages of vars
ema = tf.train.ExponentialMovingAverage(decay=0.99)
update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

# Global Step
global_step = tf.contrib.framework.get_or_create_global_step()
incr_global_step = tf.assign(global_step, global_step+1)


def convert(image):
    ''' Converts NN output back to normal image. '''
    image = image + 1 / 2 # [-1, 1] => [0, 1]
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

# Reverse processing on images so they can be outputted
converted_inputs = convert(tst_inputs)
converted_targets = convert(tst_targets)
converted_outputs = convert(tst_outputs)

# Gets image data for inputs, targets, and outputs
display_fetches = {
    "paths": tst_paths,
    "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
    "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
    "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
}

train = tf.group(update_losses, incr_global_step, gen_train)

MAX_EPOCHS = 200
OUTPUT_FREQ = 50
SAVE_FREQ = 5000
saver = tf.train.Saver(max_to_keep=1)

def save_images(fetches):
    image_dir = os.path.join('out', "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

sv = tf.train.Supervisor(logdir='out', saver=None)
start = time.time()
with sv.managed_session() as sess:
    max_steps = steps_per_epoch * MAX_EPOCHS
    start = time.time()

    for step in range(max_steps):
        def should(freq):
            return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

        fetches = {
            "train": train,
            "global_step": sv.global_step,
        }

        if should(OUTPUT_FREQ):
            fetches["discrim_loss"] = discrim_loss
            fetches["gen_loss_GAN"] = gen_loss_GAN
            fetches["gen_loss_L1"] = gen_loss_L1

        results = sess.run(fetches)

        if should(OUTPUT_FREQ):
            # global_step will have the correct step count if we resume from a checkpoint
            train_epoch = math.ceil(results["global_step"] / steps_per_epoch)
            train_step = (results["global_step"] - 1) % steps_per_epoch + 1
            rate = (step + 1) * BATCH_SIZE / (time.time() - start)
            remaining = (max_steps - step) * BATCH_SIZE / rate
            print(f"progress  epoch {train_epoch}  step {train_step}  image/sec {rate:0.1f}  remaining {remaining}m")
            print("discrim_loss", results["discrim_loss"])
            print("gen_loss_GAN", results["gen_loss_GAN"])
            print("gen_loss_L1", results["gen_loss_L1"])

        if should(SAVE_FREQ):
            print("saving model")
            saver.save(sess, os.path.join('out', "model"), global_step=sv.global_step)

        if sv.should_stop():
            break

    for step in range(tst_steps_per_epoch):
        results = sess.run(display_fetches)
        filesets = save_images(results)