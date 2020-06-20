# -*- coding: utf-8 -*-
from abc import ABC
import numpy as np
from .base import BaseTrain


class Trainer(BaseTrain, ABC):
    def __init__(self, sess, model, config, logger, data):
        super().__init__(sess, model, config, logger, data)
        self.config = config

    def train_epoch(self, cur_epoch):
        losses = []
        accs = []

        for i in range(self.config.iterate_per_epoch):
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        loss = np.mean(losses)
        acc = np.mean(accs)

        eval_acc, eval_loss = self.eval(cur_epoch, cur_it)

        summaries_dict = {
            'loss': loss,
            'acc': acc,
            'eval_acc': eval_acc,
            'eval_loss': eval_loss
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

        print(f'Average loss at epoch {cur_epoch}: {loss}')
        print(f'Train accuracy at epoch {cur_epoch}: {acc} ')

        print(f'Average validation loss at epoch {cur_epoch}: {eval_loss}')
        print(f'Validation accuracy at epoch {cur_epoch}: {eval_acc}')

    def train_step(self):
        batch_x, batch_y = self.data.get_batch()
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        _, loss, acc, step = self.sess.run(
            [self.model.train_step,
             self.model.loss,
             self.model.accuracy,
             self.model.global_step_inc
             ],
            feed_dict=feed_dict)

        if (step + 1) % self.config.skip_step == 0:
            print('Loss at step {0}: {1}'.format(step, loss))

        return loss, acc

    def eval(self, cur_epoch, cur_it):
        losses = []
        accs = []

        for i in range(self.config.num_iter_per_eval):
            acc, loss = self.eval_step()
            losses.append(loss)
            accs.append(acc)

        acc = np.mean(accs)
        loss = np.mean(losses)

        return acc, loss

    def eval_step(self):
        batch_x, batch_y = self.data.get_batch(trainable=False)

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        acc, loss = self.sess.run(
            [self.model.accuracy,
             self.model.loss],
            feed_dict=feed_dict)

        return acc, loss
