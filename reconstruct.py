import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from ae_model import *
from tensorflow.examples.tutorials.mnist import input_data


def weights_to_grid(weights, rows, cols):
    """convert the weights tensor into a grid for visualization"""
    height, width, in_channel, out_channel = weights.shape
    padded = np.pad(weights, [(1, 1), (1, 1), (0, 0), (0, rows * cols - out_channel)],
                    mode='constant', constant_values=0)
    transposed = padded.transpose((3, 1, 0, 2))
    reshaped = transposed.reshape((rows, -1))
    grid_rows = [row.reshape((-1, height + 2, in_channel)).transpose((1, 0, 2)) for row in reshaped]
    grid = np.concatenate(grid_rows, axis=0)

    return grid.squeeze()


mnist=input_data.read_data_sets('../Feature_extraction/MNIST_data',one_hot=True)
learning_rate=1e-5

x=tf.placeholder(dtype=tf.float32,shape=[None,784])
x_image = tf.reshape(x,shape=[-1, 28, 28, 1])
rec=autoencoder_model2(x_image)
# loss = tf.reduce_mean(tf.square(tf.subtract(rec, x_image)))
loss=tf.nn.l2_loss(x_image-rec)
opt=tf.train.AdamOptimizer(learning_rate).minimize(loss)

n_epochs = 1
batch_size=36
tf.summary.scalar("loss", loss)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('../Feature_extraction/logs/test-model.ckpt-29900.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('../Feature_extraction/logs'))

  # visualize results
  x_test, y_test = mnist.test.next_batch(batch_size)
  org, recon = sess.run((x, rec), feed_dict={x: x_test})

  org1=org.reshape(-1,28,28,1)
  input_images = weights_to_grid(org1.transpose((1, 2, 3, 0)), 6, 6)
  recon_images = weights_to_grid(recon.transpose((1, 2, 3, 0)), 6, 6)

  fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
  ax0.imshow(input_images, cmap=plt.cm.gray, interpolation='nearest')
  ax0.set_title('input images')
  ax1.imshow(recon_images, cmap=plt.cm.gray, interpolation='nearest')
  ax1.set_title('reconstructed images')
  plt.show()
