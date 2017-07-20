import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time


Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

BETA1 = 0.5
LR = 0.0002

EPS = 1e-12
CROP_SIZE = 256

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


SCALE_SIZE = 286
BATCH_SIZE = 1
def load_examples(input_dir):
    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input[:,:width//2,:])
        b_images = preprocess(raw_input[:,width//2:,:])

    inputs, targets = [b_images, a_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [SCALE_SIZE, SCALE_SIZE], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if SCALE_SIZE > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif SCALE_SIZE < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=BATCH_SIZE)
    steps_per_epoch = int(math.ceil(len(input_paths) / BATCH_SIZE))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


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

GAN_WEIGHT = 1.0
L1_WEIGHT = 100.0
def create_model(inputs, targets):
    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * GAN_WEIGHT + gen_loss_L1 * L1_WEIGHT

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(LR, BETA1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(LR, BETA1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(OUT_DIR, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(OUT_DIR, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

MAX_EPOCHS = 200
OUT_DIR = "plz_work"
def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    examples = load_examples('data/pix2pix/facades/train')
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    #####
    with tf.name_scope("test_images"):
        tst_examples = load_examples('data/pix2pix/facades/val/')
        # undo colorization splitting on images that we use for display/output
        tst_inputs = deprocess(tst_examples.inputs)
        tst_targets = deprocess(tst_examples.targets)
        tst_outputs = deprocess(tst_examples.outputs)

        # reverse any processing on images so they can be written to disk or displayed to user
        with tf.name_scope("convert_inputs"):
            tst_converted_inputs = convert(tst_inputs)

        with tf.name_scope("convert_targets"):
            tst_converted_targets = convert(tst_targets)

        with tf.name_scope("convert_outputs"):
            tst_converted_outputs = convert(tst_outputs)

        with tf.name_scope("encode_images"):
            tst_display_fetches = {
                "paths": examples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, tst_converted_inputs, dtype=tf.string, name="tst_input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, tst_converted_targets, dtype=tf.string, name="tst_target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, tst_converted_outputs, dtype=tf.string, name="tst_output_pngs"),
            }

    ####

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)
    logdir = OUT_DIR
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        max_steps = examples.steps_per_epoch * MAX_EPOCHS

        # training
        start = time.time()

        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            #if should(a.trace_freq):
            #    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #    run_metadata = tf.RunMetadata()

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            PROG_F = 50
            if should(PROG_F):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1

            SUMMARY_F = 0
            if should(SUMMARY_F):
                fetches["summary"] = sv.summary_op

            DISP_F = 0
            if should(DISP_F):
                fetches["display"] = display_fetches

            results = sess.run(fetches, options=options, run_metadata=run_metadata)

            if should(SUMMARY_F):
                print("recording summary")
                sv.summary_writer.add_summary(results["summary"], results["global_step"])


            if should(DISP_F):
                print("saving display images")
                filesets = save_images(results["display"], step=results["global_step"])
                append_index(filesets, step=True)

            #if should(a.trace_freq):
            #    print("recording trace")
            #    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(PROG_F):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * BATCH_SIZE / (time.time() - start)
                remaining = (max_steps - step) * BATCH_SIZE / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])

            SAVE_F = 5000
            if should(SAVE_F):
                print("saving model")
                saver.save(sess, os.path.join(OUT_DIR, "model"), global_step=sv.global_step)

            if sv.should_stop():
                break
        # testing
        #tst_examples = load_examples('data/pix2pix/facades/val/')
        # at most, process the test data once

        max_steps = min(examples.steps_per_epoch, max_steps)
        for step in range(max_steps):
            results = sess.run(tst_display_fetches)
            filesets = save_images(results)
            for i, f in enumerate(filesets):
                print("evaluated image", f["name"])
            index_path = append_index(filesets)

        print("wrote index at", index_path)


main()
