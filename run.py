# -*- coding: utf-8 -*-
import logging
import tensorflow as tf
from cnf.argument import options
from cnf.config import p_config
from cnf.log import Logger
from data_loader.opendata import DataLoad
from models.vgg16 import VGG16
from trainer.train import Trainer
from data_loader.opendata import generate_complete_data

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        data = DataLoad
        model = VGG16(config=p_config)
        logger = Logger(sess=sess, config=p_config)

        trainer = Trainer(
            sess=sess,
            model=model,
            data=data,
            config=p_config,
            logger=logger)

        trainer.train()


if __name__ == '__main__':
    p = options()
    args = p.parse_args()

    if args.generate is not None:
        generate_complete_data(args.generate)


