# -*- coding: utf-8 -*-
from argparse import ArgumentParser

description = f"""Product Detection Shopee Code League"""
parser = ArgumentParser(description=description, epilog="_" * 70)


def options() -> ArgumentParser:
    parser.add_argument('-m', '--model', help="model name, avalaible ['vgg16', 'xception', 'inceptionresnetv2', 'efficientnetb6', 'efficientnetb7', 'efficientnetl2']", default=None)
    parser.add_argument('-sm', '--savedmodel', help="path for saved model", default=None)
    return parser
