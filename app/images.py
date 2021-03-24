import glob
import imageio
import matplotlib.pyplot as plt

def generate_gif():
  anim_file = 'app/saves/img/dcgan.gif'

  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('app/saves/img/image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('app/saves/img/image_at_epoch_{:04d}.png'.format(epoch))