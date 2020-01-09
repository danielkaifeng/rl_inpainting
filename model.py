import tensorflow as tf
import numpy as np
import math

def batch_norm(x, phase_train, scope):
    n_out = x.get_shape().as_list()[-1]
    #with tf.variable_scope(scope):
    if 1:
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def leaky_relu(x, leak=0.1, name='leaky_relu'):
        return tf.maximum(x, x * leak, name=name)

class build_network():
    def __init__(self, learning_rate, size):
        self.size = size
        self.x = tf.placeholder(tf.float32, shape=[None, size, size, 3], name='x')
        self.label = tf.placeholder(tf.int64, shape=[None], name="label")

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.logits = self.encoder(self.x)

        #self.reconst_loss = tf.reduce_mean(tf.squared_difference(self.x, self.x_))
        #mean, variance = tf.nn.moments(self.feature, axes=0)
        #self.reconst_loss +=  tf.reduce_mean(tf.square(variance -1) + tf.square(mean))
        #self.D_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.zeros_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        #self.G_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))

        y = tf.one_hot(self.label, depth=2)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits=self.logits))
        #self.mse_loss = tf.reduce_mean(tf.squared_difference(self.label, self.logits))
        self.prob = tf.nn.softmax(self.logits, name='prob')
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=1), self.label), dtype=tf.float32))

        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def encoder(self, x):
            #x = tf.reshape(x, [-1, 128, 128, 3])
            #net = tf.map_fn(lambda im: tf.image.random_flip_left_right(im), x)
            x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.batch_normalization(x)
            x = tf.layers.dropout(x, 0.2)
            x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.batch_normalization(x)
            x = tf.layers.dropout(x, 0.2)
            x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.batch_normalization(x)
            x = tf.layers.dropout(x, 0.2)
            x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.batch_normalization(x)
            x = tf.layers.dropout(x, 0.2)

            #x = tf.contrib.layers.conv2d_transpose(x, 64, 3, stride=2, padding='same',
            #                activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
            #x = tf.layers.dropout(x, 0.2)
            """
            x = tf.contrib.layers.conv2d_transpose(x, 32, 3, stride=2, padding='same',
                            activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
            x = tf.layers.dropout(x, 0.2)
            x = tf.contrib.layers.conv2d_transpose(x, 32, 3, stride=2, padding='same',
                            activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
            x = tf.layers.dropout(x, 0.2)
            """
            #x = tf.contrib.layers.conv2d_transpose(x, 3, 3, stride=2, padding='same',
            #                activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
            #x = batch_norm(x, self.is_train, "bn2")
            #x = tf.layers.dropout(x, 0.2)

            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 128)
            x = tf.layers.dense(x, 2)

            #x = tf.nn.sigmoid(x)
            return x







