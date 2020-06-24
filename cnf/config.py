# -*- coding: utf-8 -*-
import os

# Basic config for foldering
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'shopee-product-detection-dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train/train/')
TEST_DIR = os.path.join(DATA_DIR, 'test/test/')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'save_models')
TRAIN_VERSION = 'version-1'


# Config for training
IMAGE_SIZE = 224        # Image size for input training
IMAGE_CHANNEL = 3       # Image channel
BATCH_SIZE = 100         # How much image will train in a batch
EPOCH_NUM = 10          # Total epoch for training
TEST_SIZE = 0.20        # Composition train and test data
LEARNING_RATE = 1e-4    # Learning rate
DECAY = 1e-6            # Still don't know
CLASS_NUM = 42

