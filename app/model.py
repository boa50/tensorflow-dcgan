import os
import tensorflow as tf
from tensorflow.keras import layers
import time

from images import generate_and_save_images

class Model:
    def __init__(self, BATCH_SIZE, MODEL_INPUT):
        self.BATCH_SIZE = BATCH_SIZE
        self.MODEL_INPUT = MODEL_INPUT

        self.noise_dim = 100
        self.num_examples_to_generate = 16
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

        self.discriminator = self.make_discriminator_model()
        self.generator = self.make_generator_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.checkpoint_dir = 'app/saves/checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generator=self.generator,
                                    discriminator=self.discriminator)

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((4, 4, 256)))
        assert model.output_shape == (None, 4, 4, 256)

        model.add(layers.Conv2DTranspose(256, (4, 4), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 4, 4, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 64, 64, 3)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.GaussianNoise(0.2))
        model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=self.MODEL_INPUT))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(64, (4, 4), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, (4, 4), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2D(256, (4, 4), strides=(1, 1), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs):
        rgb = False
        if self.MODEL_INPUT[2] == 3:
            rgb = True

        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            generate_and_save_images(self.generator, epoch + 1, self.seed, rgb=rgb)

            if (epoch + 1) % 10 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

            # generate_and_save_images(self.generator, epochs, self.seed, rgb=rgb)

    def checkpoint_restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))