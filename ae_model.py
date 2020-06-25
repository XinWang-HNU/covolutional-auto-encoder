import tensorflow as tf

Conv2D=tf.keras.layers.Conv2D
TransConv2D=tf.keras.layers.Conv2DTranspose
Maxpool2D=tf.keras.layers.MaxPool2D
Dense=tf.keras.layers.Dense
Unpool=tf.keras.layers.UpSampling2D
BatchNormalization=tf.keras.layers.BatchNormalization
relu=tf.keras.activations.relu
sigmoid=tf.keras.activations.sigmoid
Flatten=tf.keras.layers.Flatten

def autoencoder_model1(x):
    with tf.variable_scope('encoder') :

        #layer 1
        layer_1=tf.keras.layers.Conv2D(filters=2,
                                       kernel_size=[2,2],
                                       strides=[1,1],
                                       padding='Same',
                                       activation=None,
                                       name='layer_1')(x)

        batch_norm_1=tf.keras.layers.BatchNormalization(axis=-1,
                                                   momentum=0.9,
                                                   epsilon=0.001,
                                                   trainable=True,
                                                   name='layer_1/batch_norm_1')(layer_1)
        layer_1_n=tf.keras.activations.relu(batch_norm_1)

        # layer 2
        layer_2 = tf.keras.layers.Conv2D(filters=4,
                                         kernel_size=[2, 2],
                                         strides=[2, 2],
                                         activation=None,
                                         name='layer_1')(layer_1_n)

        batch_norm_2 = tf.keras.layers.BatchNormalization(axis=-1,
                                                          momentum=0.9,
                                                          epsilon=0.001,
                                                          trainable=True,
                                                          name='layer_2/batch_norm_2')(layer_2)
        layer_2_n = tf.keras.activations.relu(batch_norm_2)

        # layer 3
        layer_3 = tf.keras.layers.Conv2D(filters=8,
                                         kernel_size=[2, 2],
                                         strides=[2, 2],
                                         activation=None,
                                         name='layer_1')(layer_2_n)

        batch_norm_3 = tf.keras.layers.BatchNormalization(axis=-1,
                                                          momentum=0.9,
                                                          epsilon=0.001,
                                                          trainable=True,
                                                          name='layer_2/batch_norm_2')(layer_3)
        layer_3_n = tf.keras.activations.relu(batch_norm_3)

        # layer 4
        layer_4 = tf.keras.layers.Conv2D(filters=16,
                                         kernel_size=[2, 2],
                                         strides=[2, 2],
                                         activation=None,
                                         name='layer_1')(layer_3_n)

        batch_norm_4 = tf.keras.layers.BatchNormalization(axis=-1,
                                                          momentum=0.9,
                                                          epsilon=0.001,
                                                          trainable=True,
                                                          name='layer_2/batch_norm_2')(layer_4)
        layer_4_n = tf.keras.activations.relu(batch_norm_4)

        # layer 4
        layer_5 = tf.keras.layers.Conv2D(filters=32,
                                         kernel_size=[2, 2],
                                         strides=[2, 2],
                                         activation=None,
                                         name='layer_1')(layer_4_n)

        batch_norm_5 = tf.keras.layers.BatchNormalization(axis=-1,
                                                          momentum=0.9,
                                                          epsilon=0.001,
                                                          trainable=True,
                                                          name='layer_2/batch_norm_2')(layer_5)
        layer_5_n = tf.keras.activations.relu(batch_norm_5)

        flatten_x = tf.keras.layers.Flatten()(layer_5_n)
        fc = tf.keras.layers.Dense(20, activation=None)(flatten_x)

    with tf.variable_scope('decoder'):

        #fc
        f_x = tf.keras.layers.Dense(32, activation=None)(fc)
        b5 = tf.keras.layers.BatchNormalization(axis=-1,
                                                momentum=0.9,
                                                epsilon=0.001,
                                                trainable=True,
                                                name='f_x/batch_norm_13')(f_x)
        fc_n = tf.keras.activations.relu(b5)
        l5_n = tf.reshape(fc_n, [-1, 1, 1, 32])

        # layer 4
        l4 = tf.keras.layers.Conv2DTranspose(filters=16,
                                               kernel_size=[3, 3],
                                               strides=[2, 2],
                                               activation=None,
                                               name='layer_12')(l5_n)
        b4 = tf.keras.layers.BatchNormalization(axis=-1,
                                                           momentum=0.9,
                                                           epsilon=0.001,
                                                           trainable=True,
                                                           name='layer_12/batch_norm_12')(l4)
        l4_n = tf.keras.activations.relu(b4)

        # layer 3
        l3 = tf.keras.layers.Conv2DTranspose(filters=8,
                                             kernel_size=[3, 3],
                                             strides=[2, 2],
                                             activation=None,
                                             name='layer_12')(l4_n)
        b3 = tf.keras.layers.BatchNormalization(axis=-1,
                                                momentum=0.9,
                                                epsilon=0.001,
                                                trainable=True,
                                                name='layer_12/batch_norm_12')(l3)
        l3_n = tf.keras.activations.relu(b3)

        # layer 2
        l2 = tf.keras.layers.Conv2DTranspose(filters=4,
                                             kernel_size=[2, 2],
                                             strides=[2, 2],
                                             activation=None,
                                             name='layer_12')(l3_n)
        b2 = tf.keras.layers.BatchNormalization(axis=-1,
                                                momentum=0.9,
                                                epsilon=0.001,
                                                trainable=True,
                                                name='layer_12/batch_norm_12')(l2)
        l2_n = tf.keras.activations.relu(b2)

        # layer 1
        l1 = tf.keras.layers.Conv2DTranspose(filters=2,
                                             kernel_size=[2, 2],
                                             strides=[2, 2],
                                             activation=None,
                                             name='layer_12')(l2_n)
        b1 = tf.keras.layers.BatchNormalization(axis=-1,
                                                momentum=0.9,
                                                epsilon=0.001,
                                                trainable=True,
                                                name='layer_12/batch_norm_12')(l1)
        l1_n = tf.keras.activations.relu(b1)

        # layer 0
        l0 = tf.keras.layers.Conv2DTranspose(filters=1,
                                             kernel_size=[1, 1],
                                             strides=[1, 1],
                                             activation=None,
                                             name='layer_12')(l1_n)
        b0 = tf.keras.layers.BatchNormalization(axis=-1,
                                                momentum=0.9,
                                                epsilon=0.001,
                                                trainable=True,
                                                name='layer_12/batch_norm_12')(l0)
        rec = tf.keras.activations.relu(b0)

        return rec


def autoencoder_model2(x):
    with tf.variable_scope('encoder'):
        conv1 = Conv2D(filters=64, kernel_size=[5, 5], strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv1')(x)
        pool1 = Maxpool2D(pool_size=[2, 2], strides=[2, 2],name='pool1')(conv1)
        conv2 = Conv2D(filters=32, kernel_size=[5, 5], strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv2')(pool1)
        pool2 = Maxpool2D(pool_size=[2, 2], strides=[2, 2],name='pool2')(conv2)
        un_fold = tf.keras.layers.Flatten(name='un_fold')(pool2)
        encoded = tf.keras.layers.Dense(20, activation=tf.nn.relu,name='encoded')(un_fold)
    with tf.variable_scope('decoder'):
        decoded = Dense(7 * 7 * 32, activation=tf.nn.relu,name='decoded')(encoded)
        fold = tf.reshape(decoded, shape=[-1, 7, 7, 32],name='fold')
        unpool2 = Unpool(name='unpool2')(fold)
        Deconv2 = TransConv2D(filters=64, kernel_size=[5, 5], strides=[1, 1], padding='same', activation=tf.nn.relu,
                              name='deconv2')(unpool2)
        unpool1 = Unpool(name='unpool1')(Deconv2)
        Deconv1 = TransConv2D(filters=1, kernel_size=[5, 5], strides=[1, 1], padding='same', activation=tf.nn.sigmoid,
                           name='Deconv1')(unpool1)

        return Deconv1

def autoencoder_model3(x):

    with tf.variable_scope('encoder'):
        conv1 =Conv2D(filters=64, kernel_size=[5, 5],strides=[1, 1],padding='same',activation=None,name='conv1')(x)
        batch_norm_1 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,trainable=True,name='batch_norm_1')(conv1)
        with tf.name_scope('relu_1'):
            layer_1_n = relu(batch_norm_1)
        pool1 = Maxpool2D(pool_size=[2, 2], strides=[2, 2],name='pool1')(layer_1_n)

        conv2 =Conv2D(filters=32,kernel_size=[5, 5],strides=[1, 1],padding='same',activation=None,name='conv2')(pool1)
        batch_norm_2 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,trainable=True,name='batch_norm_2')(conv2)
        with tf.name_scope('relu_2'):
            layer_2_n = relu(batch_norm_2)
        pool2 = Maxpool2D(pool_size=[2, 2], strides=[2, 2],name='pool2')(layer_2_n)

        un_fold = Flatten(name='un_fold')(pool2)

        encoded = Dense(20, activation=None,name='encoded')(un_fold)
        batch_norm_3 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, trainable=True, name='batch_norm_3')(encoded)
        with tf.name_scope('relu_3'):
            latent = relu(batch_norm_3)

    with tf.variable_scope('decoder'):

        decoded = Dense(7 * 7 * 32, activation=None,name='decoded')(latent)
        batch_norm_4 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, trainable=True, name='batch_norm_4')(decoded)
        with tf.name_scope('relu_4'):
            latent_ = relu(batch_norm_4)

        fold = tf.reshape(latent_, shape=[-1, 7, 7, 32],name='fold')

        unpool2 = Unpool(name='unpool2')(fold)
        Deconv2 = TransConv2D(filters=64, kernel_size=[5, 5], strides=[1, 1], padding='same', activation=None,name='deconv2')(unpool2)
        batch_norm_5 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, trainable=True, name='batch_norm_5')(Deconv2)
        with tf.name_scope('relu_5'):
            layer_2_n_ = relu(batch_norm_5)

        unpool1 = Unpool(name='unpool1')(layer_2_n_)
        Deconv1 = TransConv2D(filters=1, kernel_size=[5, 5], strides=[1, 1], padding='same', activation=None,name='deconv1')(unpool1)
        batch_nrm_6 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, trainable=True, name='batch_norm_5')(Deconv1)
        with tf.name_scope('relu_6'):
            layer_1_n_ = sigmoid(batch_nrm_6)

        return layer_1_n_
