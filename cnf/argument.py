# -*- coding: utf-8 -*-
from argparse import ArgumentParser

description = f"""Product Detection Shopee Code League"""
parser = ArgumentParser(description=description, epilog="_" * 70)


def options() -> ArgumentParser:
    parser.add_argument('-m', '--model', help="model name, avalaible ['vgg16']", default=None)
    parser.add_argument('-f', '--filemodel', help="model filename", default=None)
    return parser
