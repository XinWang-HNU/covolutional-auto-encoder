from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import numpy as np
from ae_model import *

import tensorflow as tf

mnist=input_data.read_data_sets('../Feature_extraction/MNIST_data',one_hot=True)

save_dir='../Feature_extraction/Mnist_Picture'

learning_rate=1e-5
x=tf.placeholder(dtype=tf.float32,shape=[None,784])
x_image = tf.reshape(x,shape=[-1, 28, 28, 1])
rec=autoencoder_model3(x_image)

loss = tf.reduce_mean(tf.square(tf.subtract(rec, x_image)))
# loss=tf.nn.l2_loss(x_image-rec)
opt=tf.train.AdamOptimizer(learning_rate).minimize(loss)

logs_path='/Users/ziqiangpu/Downloads/Feature_extraction/logs'
n_epochs = 1
batch_size=100
tf.summary.scalar("loss", loss)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
passes=30000
saver=tf.train.Saver()
ckpt_path = '../Feature_extraction/logs/test-model.ckpt'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(n_epochs):
        n_batches = int(len(mnist.train.images) / batch_size)
        # Loop over all batches
        for i in range(passes):
            batch_x, _ = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, summary = sess.run([opt,  merged_summary_op], feed_dict={x: batch_x})
            # Compute average loss
            if i % 100 == 0:
                c = loss.eval(feed_dict={x: batch_x})
                print('epoch {}, training loss {}'.format(i,c))
            if i % 1000==0:
                writer.add_summary(summary)
            if i % 1000 == 0:
                 # write log
                save_path=saver.save(sess,ckpt_path,global_step=i)
                print('checkpoint saved in %s'% save_path)

    print('Optimization Finished')
    print('Cost:', loss.eval({x: mnist.test.images}))








