# -*- coding: utf-8 -*-
import tensorflow as tf
from cnf.config import EPOCH_NUM


class BaseTrain:
    def __init__(self, sess, model, logger, data):
        self.sess = sess
        self.model = model
        self.logger = logger
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        cur_epoch_tensor = self.model.cur_epoch_tensor.eval(self.sess)
        for cur_epoch in range(cur_epoch_tensor, EPOCH_NUM + 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, cur_epoch):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
