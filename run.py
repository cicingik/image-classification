# -*- coding: utf-8 -*-
import logging
import sys
import tensorflow as tf
from cnf.argument import options
from cnf.log import Logger
from data_loader.data_loader import DataLoader
from models.vgg16 import VGG16
from trainer.train import Trainer
from data_loader.opendata import generate_complete_data

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main(data_set: str):
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        data = DataLoader(data_set)
        model = VGG16()
        logger = Logger(sess=sess)

        trainer = Trainer(
            sess=sess,
            model=model,
            data=data,
            logger=logger)

        trainer.train()


if __name__ == '__main__':
    p = options()
    args = p.parse_args()

    if args.generate is not None:
        generate_complete_data(args.generate)

    if not all([args.train]):
        p.print_help()
        sys.exit(2)

    main(args.train)


