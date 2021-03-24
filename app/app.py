import tensorflow as tf

from model import Model

# Configurações para o tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Preparação do dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

EPOCHS = 50
BUFFER_SIZE = 60000
BATCH_SIZE = 512

train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
    .shuffle(BUFFER_SIZE) \
    .batch(BATCH_SIZE)

# Treinamento do modelo
model = Model(BATCH_SIZE)
model.train(train_dataset, EPOCHS)

# model.checkpoint_restore()