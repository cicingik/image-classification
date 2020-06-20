# -*- coding: utf-8 -*-
import os
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'shopee-product-detection-dataset')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
IMAGE_SIZE = 600
JSON_CONFIG = '/Users/mymac/cingik/product-detection/image-classification/cnf/train_config.json'


def process_config() -> dict:
    with open(JSON_CONFIG, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict


p_config = process_config()



