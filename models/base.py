# -*- coding: utf-8 -*-
import tensorflow as tf
from cnf.config import CHECKPOINT


class BaseModel:
    def __init__(self):
        self.global_step_tensor = None
        self.cur_epoch_tensor = None
        self.increment_cur_epoch_tensor = None
        self.saver = tf.train.Saver
        self.init_cur_epoch()

    def save(self, sess):
        self.saver.save(sess, CHECKPOINT, self.global_step_tensor)

    def load(self, sess):
        last_cp = tf.train.latest_checkpoint(CHECKPOINT)
        if last_cp:
            self.saver.restore(sess, CHECKPOINT)

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError


def is_retrain(retrain: str, name: str) -> bool:
    if retrain == 'complete':
        rt_status = True
    elif retrain == 'semi':
        if name in ['conv3', 'conv4', 'conv5']:
            rt_status = True
        else:
            rt_status = False
    else:
        rt_status = False

    return rt_status
