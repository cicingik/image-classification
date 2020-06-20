# -*- coding: utf-8 -*-
from argparse import ArgumentParser

description = f"""Product Detection Shopee Code League"""
parser = ArgumentParser(description=description, epilog="_" * 70)


def options() -> ArgumentParser:
    parser.add_argument('-g', '--generate', help="path to save generate complete data file", default=None)
    parser.add_argument('-t', '--train', help="path from complete data file", default=None)
    return parser
