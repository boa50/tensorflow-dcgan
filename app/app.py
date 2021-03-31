import tensorflow as tf

from model import Model

# Configurações para o tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

EPOCHS = 100
BATCH_SIZE = 128

# # Preparação do dataset MNIST
# BUFFER_SIZE = 60000
# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5

# train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
#     .shuffle(BUFFER_SIZE) \
#     .batch(BATCH_SIZE)


# Preparação de dataset local

# Celeb Faces (https://www.kaggle.com/jessicali9530/celeba-dataset)
# imgs_path = 'app/dataset/archive/img_align_celeba/'

# Anime Faces (https://github.com/bchao1/Anime-Face-Dataset)
# imgs_path = 'app/dataset/data/'

# Anime Faces v2 (https://www.kaggle.com/scribbless/another-anime-face-dataset)
imgs_path = 'app/dataset/anime_faces_v2/'

IMAGE_SIZE = (64, 64)
MODEL_INPUT = [64, 64, 3]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    imgs_path,
    label_mode=None,
    seed=50,
    color_mode='rgb',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
train_ds = train_ds.map(lambda x: (normalization_layer(x)))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# import matplotlib.pyplot as plt
# import numpy as np
# for images in train_ds.take(1):
#     print(images.shape)
#     print(images[0].numpy())
#     plt.imshow((images[0] * 127.5 + 127.5).numpy().astype('uint8'))
#     plt.savefig('app/saves/img/atest.png')
#     break

# Treinamento do modelo
model = Model(BATCH_SIZE, MODEL_INPUT)

# model.generator.summary()
# model.discriminator.summary()

# from images import generate_and_save_images
# seed = tf.random.normal([16, 100])
# generate_and_save_images(model.generator, 0, seed, rgb=True)

model.train(train_ds, EPOCHS)

# model.checkpoint_restore()