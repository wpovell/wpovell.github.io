import tensorflow as tf
import os
import glob
import math
import random
import time

class Pix2Pix:
    # Num of gen filters
    NGF = 64
    # Num discrim filters
    NDF = 64

    # Size to scale to
    SCALE_SIZE = 286
    # Size to crop to after scaling
    CROP_SIZE = 256

    A_TO_B = False
    OUT_DIR = 'out'

    # Weights for the two GAN losses
    GAN_WEIGHT = 1.
    L1_WEIGHT = 100.

    # Optimizer params
    LR = 0.0002
    BETA1 = 0.5

    # Val used to prevent 0 erros
    EPS = 1e-12

    # How often to print progress/save
    PROG_F = 50
    SAVE_F = 5000

    MAX_EPOCHS = 200
    BATCH_SIZE = 1

    def __init__(self, data_dir):
        """ Loads in data & initialized model. """
        print("Loading data")
        self._load_data(data_dir)
        print("Building model")
        self._build_model()

    def _process_image(self, image, seed):
        """ Process image to be inputted to model.
         Scales pixels to [-1, 1] and crops to CROP_SIZE from SCALE_SIZE randomly.
         """
        # [0, 1] => [-1, 1]
        image = image * 2 - 1

        # Scale down to SCALE_SIZE
        image = tf.image.resize_images(image, [self.SCALE_SIZE, self.SCALE_SIZE], method=tf.image.ResizeMethod.AREA)

        # Choose random offset from corner and crop to CROP_SIZE from it
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, self.SCALE_SIZE - self.CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(image, offset[0], offset[1], self.CROP_SIZE, self.CROP_SIZE)

        return image

    def _load_data(self, path):
        """ Loads data from path. """

        if not os.path.exists(path):
            raise Exception("No such directory: {}".format(path))

        # All jpgs
        input_paths = glob.glob(os.path.join(path, '*.jpg'))

        # Data pipeline
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_image = tf.image.decode_jpeg(contents)
        raw_image = tf.image.convert_image_dtype(raw_image, dtype=tf.float32)
        raw_image.set_shape([None, None, 3])

        # Split image into left and right side
        width = tf.shape(raw_image)[1]
        left, right = raw_image[:, :width // 2, :], raw_image[:, width // 2:, :]

        # Process image
        seed = random.randint(0, 2 ** 31 - 1)
        left = self._process_image(left, seed)
        right = self._process_image(right, seed)

        # Assign sides to input/target
        if self.A_TO_B:
            inputs, targets = left, right
        else:
            inputs, targets = right, left

        # Create batches
        self.paths, self.inputs, self.targets = tf.train.batch([paths, inputs, targets], batch_size=self.BATCH_SIZE)
        self.steps_per_epoch = int(math.ceil(len(input_paths) / self.BATCH_SIZE))

    @staticmethod
    def _conv(batch_input, out_channels, stride):
        """ Convolve. """
        with tf.variable_scope("conv"):
            in_channels = batch_input.get_shape()[3]
            conv_filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(0, 0.02))
            # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
            #     => [batch, out_height, out_width, out_channels]
            padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            conv = tf.nn.conv2d(padded_input, conv_filter, [1, stride, stride, 1], padding="VALID")
            return conv

    @staticmethod
    def _lrelu(x, a):
        """ Leaky ReLU.
        -a is slope of line when x < 0.
        """
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    @staticmethod
    def _batchnorm(inp):
        """ Batch Normalization. """
        with tf.variable_scope("batchnorm"):
            channels = inp.get_shape()[3]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(1.0, 0.02))
            mean, variance = tf.nn.moments(inp, axes=[0, 1, 2], keep_dims=False)
            variance_epsilon = 1e-5
            normalized = tf.nn.batch_normalization(inp, mean, variance, offset, scale,
                                                   variance_epsilon=variance_epsilon)
            return normalized

    @staticmethod
    def _deconv(batch_input, out_channels):
        with tf.variable_scope("deconv"):
            batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
            deconv_filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(0, 0.02))
            # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
            #     => [batch, out_height, out_width, out_channels]
            deconv = tf.nn.conv2d_transpose(batch_input,
                                            deconv_filter,
                                            [batch, in_height * 2, in_width * 2, out_channels],
                                            [1, 2, 2, 1],
                                            padding="SAME")
            return deconv

    def _create_generator(self, generator_inputs):
        """ Create generator network. """
        layers = []

        # ENCODER ###

        # encoder 1: [batch, 256, 256, 3] => [batch, 128, 128, NGF]
        with tf.variable_scope("encoder_1"):
            output = self._conv(generator_inputs, self.NGF, stride=2)
            layers.append(output)

        layer_specs = [
            self.NGF * 2,  # encoder 2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.NGF * 4,  # encoder 3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.NGF * 8,  # encoder 4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.NGF * 8,  # encoder 5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.NGF * 8,  # encoder 6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.NGF * 8,  # encoder 7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.NGF * 8,  # encoder 8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_{}".format(len(layers) + 1)):
                rectified = self._lrelu(layers[-1], 0.2)
                convolved = self._conv(rectified, out_channels, stride=2)
                output = self._batchnorm(convolved)
                layers.append(output)

        # DECODER ###

        # (channels, dropout)
        layer_specs = [
            (self.NGF * 8, 0.5),  # decoder 8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (self.NGF * 8, 0.5),  # decoder 7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (self.NGF * 8, 0.5),  # decoder 6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (self.NGF * 8, 0.0),  # decoder 5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (self.NGF * 4, 0.0),  # decoder 4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (self.NGF * 2, 0.0),  # decoder 3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (self.NGF, 0.0),  # decoder 2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            # Encoder layer to concat with
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_{}".format(skip_layer + 1)):
                if decoder_layer == 0:
                    # First decoder layer doesn't have skip connections
                    inp = layers[-1]
                else:
                    inp = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(inp)
                output = self._deconv(rectified, out_channels)
                output = self._batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder 1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, 3]
        with tf.variable_scope("decoder_1"):
            inp = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(inp)
            output = self._deconv(rectified, 3)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]

    def _create_discriminator(self, discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, 3] => [batch, height, width, 6]
        inp = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer 1: [batch, 256, 256, 6] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = self._conv(inp, self.NDF, stride=2)
            rectified = self._lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer 2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer 3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer 4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_{}".format(len(layers) + 1)):
                out_channels = self.NDF * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # Last layer here has stride 1
                convolved = self._conv(layers[-1], out_channels, stride=stride)
                normalized = self._batchnorm(convolved)
                rectified = self._lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_{}".format(len(layers) + 1)):
            convolved = self._conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    def _build_model(self):
        with tf.variable_scope("generator"):
            self.outputs = self._create_generator(self.inputs)

        with tf.variable_scope("discriminator"):
            predict_real = self._create_discriminator(self.inputs, self.targets)

        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = self._create_discriminator(self.inputs, self.outputs)

        # Discriminator loss
        self.discrim_loss = tf.reduce_mean(-(tf.log(predict_real + self.EPS) + tf.log(1 - predict_fake + self.EPS)))

        # Generator loss
        gen_loss_gan = tf.reduce_mean(-tf.log(predict_fake + self.EPS))
        gen_loss_l1 = tf.reduce_mean(tf.abs(self.targets - self.outputs))
        self.gen_loss = gen_loss_gan * self.GAN_WEIGHT + gen_loss_l1 * self.L1_WEIGHT

        # Discriminator train op
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(self.LR, self.BETA1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(self.discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        # Generator train op
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(self.LR, self.BETA1)
            gen_grads_and_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([self.discrim_loss, gen_loss_gan, gen_loss_l1])

        global_step = tf.contrib.framework.get_or_create_global_step()
        self.incr_global_step = tf.assign(global_step, global_step + 1)

        self.train_op = tf.group(update_losses, self.incr_global_step, gen_train)

    @staticmethod
    def _deprocess_image(img):
        img = (img + 1) / 2
        return tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)

    def train(self):
        """ Train model. """
        max_steps = self.MAX_EPOCHS * self.steps_per_epoch
        saver = tf.train.Saver(max_to_keep=1)
        sv = tf.train.Supervisor(logdir=self.OUT_DIR, save_summaries_secs=0, saver=None)
        with sv.managed_session() as sess:
            start = time.time()

            for step in range(10):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                fetches = {
                    "train": self.train_op,
                    "global_step": sv.global_step,
                }

                if should(self.PROG_F):
                    fetches["discrim_loss"] = self.discrim_loss
                    fetches["gen_loss"] = self.gen_loss

                results = sess.run(fetches)

                if should(self.PROG_F):
                    train_epoch = math.ceil(results["global_step"] / self.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % self.steps_per_epoch + 1
                    rate = (step + 1) * self.BATCH_SIZE / (time.time() - start)
                    remaining = (max_steps - step) * self.BATCH_SIZE / rate
                    print(("progress  " +
                           "epoch {}  " +
                           "step {}  " +
                           "image/sec {:0.1f} " +
                           "remaining {}m").format(train_epoch, train_step, rate, remaining // 60))

                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss"])

                if should(self.SAVE_F):
                    print("saving model")
                    saver.save(sess, os.path.join(self.OUT_DIR, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break
            saver.save(sess, os.path.join(self.OUT_DIR, "model"), global_step=sv.global_step)

    def _save_images(self, results):
        """ Save images from results. """
        image_dir = os.path.join(self.OUT_DIR, "images")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for i, in_path in enumerate(results["paths"]):
            num, _ = os.path.splitext(os.path.basename(in_path.decode('utf8')))
            for kind in ["inputs", "outputs", "targets"]:
                filename = num + "-" + kind + ".png"
                out_path = os.path.join(image_dir, filename)
                contents = results[kind][i]
                with open(out_path, "wb") as f:
                    f.write(contents)

    def test(self):
        """ Save model output for images from data. """
        output_images = {
            "paths": self.paths,
            "inputs": tf.map_fn(tf.image.encode_png, self._deprocess_image(self.inputs),
                                dtype=tf.string, name="inputs_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, self._deprocess_image(self.targets),
                                dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, self._deprocess_image(self.outputs),
                                dtype=tf.string, name="output_pngs"),
        }

        saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=self.OUT_DIR, save_summaries_secs=0, saver=None)
        with sv.managed_session() as sess:
            # Restore from checkpoint
            checkpoint = tf.train.latest_checkpoint(self.OUT_DIR)
            saver.restore(sess, checkpoint)
            # Save outputs
            for step in range(self.steps_per_epoch):
                results = sess.run(output_images)
                self._save_images(results)


if __name__ == '__main__':
    from sys import argv

    if 'train' in argv:
        p2p = Pix2Pix('data/pix2pix/facades/train')
        print("Training")
        p2p.train()
    elif 'test' in argv:
        p2p = Pix2Pix('data/pix2pix/facades/val')
        print("Testing")
        p2p.test()
    else:
        print("Bad args")
        exit(1)
