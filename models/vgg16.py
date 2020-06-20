from abc import ABC
import numpy as np
import tensorflow as tf
from .base import BaseModel, is_retrain


class VGG16(BaseModel, ABC):
    def __init__(self, config, trainable=True, retrain='complete'):
        super(VGG16, self).__init__(config)
        self.config = config
        self.data = np.load(self.config.weights, encoding='latin1').item()
        self.trainable = trainable
        self.retrain = retrain
        self.classes = self.config.class_num
        self.logits = None
        self.logits_argmax = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        self.train_step = None
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Make input variable with tf.placeholder
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32,
                               shape=[None, self.config.image_shape, self.config.image_shape, self.config.image_channels],
                               name='x')
            self.y = tf.placeholder(tf.int32,
                               shape=[None],
                               name='y')
            tf.add_to_collection('input', self.x)
            tf.add_to_collection('input', self.y)

        # Make network based VGG16
        with tf.name_scope('network'):
            conv1_1 = self.conv_layer(self.x, filters=64, k_size=3, stride=1, padding='SAME', name='conv1_1')
            conv1_2 = self.conv_layer(conv1_1, filters=64, k_size=3, stride=1, padding='SAME', name='conv1_2')
            pool1 = self.maxpool(conv1_2, k_size=2, stride=2, padding='SAME', scope_name='pool1')

            conv2_1 = self.conv_layer(pool1, filters=128, k_size=3, stride=1, padding='SAME', name='conv2_1')
            conv2_2 = self.conv_layer(conv2_1, filters=128, k_size=3, stride=1, padding='SAME', name='conv2_2')
            pool2 = self.maxpool(conv2_2, k_size=2, stride=2, padding='SAME', scope_name='pool2')

            conv3_1 = self.conv_layer(pool2, filters=256, k_size=3, stride=1, padding='SAME', name='conv3_1')
            conv3_2 = self.conv_layer(conv3_1, filters=256, k_size=3, stride=1, padding='SAME', name='conv3_2')
            conv3_3 = self.conv_layer(conv3_2, filters=256, k_size=3, stride=1, padding='SAME', name='conv3_3')
            conv3_4 = self.conv_layer(conv3_3, filters=256, k_size=3, stride=1, padding='SAME', name='conv3_4')
            pool3 = self.maxpool(conv3_4, k_size=2, stride=2, padding='SAME', scope_name='pool3')

            conv4_1 = self.conv_layer(pool3, filters=512, k_size=3, stride=1, padding='SAME', name='conv4_1')
            conv4_2 = self.conv_layer(conv4_1, filters=512, k_size=3, stride=1, padding='SAME', name='conv4_2')
            conv4_3 = self.conv_layer(conv4_2, filters=512, k_size=3, stride=1, padding='SAME', name='conv4_3')
            conv4_4 = self.conv_layer(conv4_3, filters=512, k_size=3, stride=1, padding='SAME', name='conv4_4')
            pool4 = self.maxpool(conv4_4, k_size=2, stride=2, padding='SAME', scope_name='pool4')

            conv5_1 = self.conv_layer(pool4, filters=512, k_size=3, stride=1, padding='SAME', name='conv5_1')
            conv5_2 = self.conv_layer(conv5_1, filters=512, k_size=3, stride=1, padding='SAME', name='conv5_2')
            conv5_3 = self.conv_layer(conv5_2, filters=512, k_size=3, stride=1, padding='SAME', name='conv5_3')
            conv5_4 = self.conv_layer(conv5_3, filters=512, k_size=3, stride=1, padding='SAME', name='conv5_4')
            pool5 = self.maxpool(conv5_4, k_size=2, stride=2, padding='SAME', scope_name='pool5')

            cur_dim = pool5.get_shape()
            pool5_dim = cur_dim[1] * cur_dim[2] * cur_dim[3]
            pool5_flatten = tf.reshape(pool5, shape=[-1, pool5_dim])

            fc6 = self.fully_connected(pool5_flatten, out_dim=1024, scope_name='fc6', activation=tf.nn.relu)

            fc7 = self.fully_connected(fc6, out_dim=512, scope_name='fc7', activation=tf.nn.relu)

            self.logits = self.fully_connected(fc7, out_dim=self.classes, scope_name='logits', activation=False)

            tf.add_to_collection('logits', self.logits)

            with tf.name_scope('logits_argmax'):
                self.logits_argmax = tf.argmax(self.logits, axis=1, output_type=tf.int64, name='out_argmax')

            with tf.name_scope('loss'):
                entropy = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.logits)
                self.loss = tf.reduce_mean(entropy, name='loss')

            with tf.name_scope('train_step'):
                self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

            with tf.name_scope('accuracy'):
                prediction = tf.nn.softmax(self.logits, name='prediction')
                correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.cast(self.y, dtype=tf.int64))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.add_to_collection('train', self.train_step)
            tf.add_to_collection('loss', self.loss)
            tf.add_to_collection('acc', self.accuracy)

    def conv_layer(self, inputs, filters, k_size, stride, padding, name, active=False):
        in_channels = inputs.shape[-1]
        out_channels = filters
        rt_status = is_retrain(self.retrain, name)

        init_value_in = tf.truncated_normal(shape=[k_size, k_size, in_channels, out_channels], stddev=0.001)
        init_value_out = tf.truncated_normal([out_channels], stddev=0.001)

        kernel = self.get_var(init_value_in, name, 0, name + "_filters", retrain=rt_status)
        biases = self.get_var(init_value_out, name, 1, name + "_biases", retrain=rt_status)

        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
        conv = tf.nn.bias_add(conv, biases)

        f_conv = tf.nn.tanh(conv) if active else tf.nn.relu(conv)

        return f_conv

    def get_var(self, initial_value, name, idx, var_name, retrain=True):
        if self.data is not None and name in self.data:
            value = self.data[name][idx]
        else:
            value = initial_value

        if self.trainable and retrain:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.Variable(value, name=var_name, trainable=False)

        return var

    @staticmethod
    def maxpool(inputs, k_size, stride, padding='VALID', scope_name='maxpool'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            pool = tf.nn.max_pool(inputs, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding=padding)
        return pool

    @staticmethod
    def fully_connected(inputs, out_dim, scope_name, activation=tf.nn.relu):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            in_dim = inputs.shape[-1]
            w = tf.get_variable('weights', [in_dim, out_dim], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases', [out_dim], initializer=tf.random_normal_initializer())
            out = tf.nn.bias_add(tf.matmul(inputs, w), b)

            if activation:
                out = activation(out)
            return out

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
