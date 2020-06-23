# -*- coding: utf-8 -*-
import copy
import logging.config
import tensorflow as tf
import os
import colorama
from .config import SUMMARY_DICT

__logger = None

LOG_COLORS = {
    logging.CRITICAL: colorama.Fore.LIGHTRED_EX,
    logging.ERROR: colorama.Fore.RED,
    logging.WARNING: colorama.Fore.YELLOW,
    logging.INFO: colorama.Fore.CYAN,
    logging.DEBUG: colorama.Fore.WHITE
}


class ColorFormatter(logging.Formatter):
    def format(self, record, *args, **kwargs):
        # if the corresponding logger has children, they may receive modified
        # record, so we want to keep it intact
        new_record = copy.copy(record)
        if new_record.levelno in LOG_COLORS:
            # we want levelname to be in different color, so let's modify it
            new_record.levelname = "{color_begin}{level}{color_end}".format(
                level=new_record.levelname,
                color_begin=LOG_COLORS[new_record.levelno],
                color_end=colorama.Style.RESET_ALL,
            )
        # now we can let standart formatting take care of the rest
        return super(ColorFormatter, self).format(new_record, *args, **kwargs)


def configure_logger(filepath=None, level=logging.DEBUG):
    """configure logging"""
    level_str = logging.getLevelName(level)
    __conf = {
        'version': 1,
        'formatters': {
            'default': {
                '()': 'cnf.log.ColorFormatter',
                'format': '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'}
        },
        'handlers': {
            'console': {
                'level': f'{level_str}',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            }
        },
        'root': {
            'level': f'{level_str}',
            'handlers': ['console']
        },
        'disable_existing_loggers': False
    }

    if filepath:
        __conf['handlers'].update({
            'file': {
                'level': f'{level_str}',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': filepath,
                'maxBytes': 1024,
                'backupCount': 3
            }
        })

        # noinspection PyTypeChecker
        __conf['root'].update({
            'handlers': ['console', 'file']
        })

    logging.config.dictConfig(__conf)


class Logger:
    def __init__(self, sess):
        self.sess = sess
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(SUMMARY_DICT, "train"))
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(SUMMARY_DICT, "test"))

    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):

        train_sw = self.train_summary_writer
        test_sw = self.test_summary_writer

        summary_writer = train_sw if summarizer == "train" else test_sw

        if summaries_dict is not None:
            summary_list = []
            for tag, value in summaries_dict.items():
                if tag not in self.summary_ops:
                    if len(value.shape) <= 1:
                        self.summary_placeholders[tag] = tf.placeholder(
                            'float32', value.shape, name=tag)
                        self.summary_ops[tag] = tf.summary.scalar(
                            tag, self.summary_placeholders[tag])
                    else:
                        self.summary_placeholders[tag] = tf.placeholder(
                            'float32', [None] + list(value.shape[1:]),
                            name=tag)
                        self.summary_ops[tag] = tf.summary.image(
                            tag, self.summary_placeholders[tag])

                summary_list.append(
                    self.sess.run(
                        self.summary_ops[tag],
                        feed_dict={self.summary_placeholders[tag]: value}))

            for summary in summary_list:
                summary_writer.add_summary(summary, step)
