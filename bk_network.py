import numpy as np
import tensorflow as tf

def leaky_relu(x, leak=0.1, name='leaky_relu'):
    return tf.maximum(x, x * leak, name=name)

class agent():
    def __init__(self):
        width = 128
        height = 128
        lr = 1e-6
        lr2 = 0.001
        self.mask = tf.placeholder(tf.float32, shape=[None, width, height, 3], name='mask')
        self.reward = tf.placeholder(tf.float32, shape=[None], name="reward")
        self.reward_action = tf.placeholder(tf.float32, shape=[None, 7], name="r_action")
        self.img = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='img')
#self.mask = tf.placeholder(tf.float32, shape=[None, mask_size, mask_size, 3], name='mask')
        self.action = self.policy_net(self.mask)
        self.q_val = self.value_net(self.action, self.mask)
#self.dropout = tf.placeholder(tf.float32, name='dropout')
#self.is_train = tf.placeholder(tf.bool, name='is_train')

        #self.D_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.zeros_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        #self.G_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
#        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits=self.logits))
#       self.prob = tf.nn.softmax(self.logits, name='prob')
#self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=1), self.label), dtype=tf.float32))

        self.q_loss = tf.reduce_mean(tf.squared_difference(self.q_val, self.reward))
        self.p_loss = -self.q_val
        #ra_loss = 100 * self.reward * tf.reduce_mean(tf.squared_difference(self.reward_action, self.action), axis = 1)
        ra_loss = 100 * tf.reduce_mean(tf.squared_difference(self.reward_action, self.action), axis = 1)
        self.ra_loss = tf.reduce_mean(ra_loss)

        p_vars =  [var for var in tf.global_variables() if  "policy" in var.name]
        q_vars =  [var for var in tf.global_variables() if  "value" in var.name]
        self.q_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.q_loss)#, var_list = q_vars)
        self.p_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.p_loss, var_list = p_vars)
        self.ra_opt = tf.train.AdamOptimizer(learning_rate=lr2).minimize(self.ra_loss) #, var_list = p_vars)


    def policy_net(self, mask):
        with tf.variable_scope("policy") as scope:
            x = tf.layers.conv2d(mask, filters=64, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.batch_normalization(x)
            x = tf.layers.dropout(x, 0.2)
            x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.batch_normalization(x)
            x = tf.layers.dropout(x, 0.2)

            x = tf.layers.flatten(x)
            #x = tf.layers.dense(x, 128)
            x = tf.layers.dense(x, 7)
            action = tf.nn.sigmoid(x)
            action = tf.clip_by_value(action, 0.05, 0.95)

            return action

    def value_net(self, action, mask):
        with tf.variable_scope("value") as scope:
            x = tf.layers.conv2d(mask, filters=64, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
            x = tf.layers.batch_normalization(x)
            x = tf.layers.flatten(x)

            ax = tf.layers.dense(action, 128)
            #ax = tf.layers.dense(action, 64)
            x = tf.concat([x,ax], axis=1)
            x = tf.layers.dense(x, 64)
            q_val = tf.layers.dense(x, 1)
            return q_val

if __name__ == "__main__":
    a = agent()
    action = a.policy(1)
    print(action)




