# -*- coding: utf-8 -*-
import os


# Basic config for foldering
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'shopee-product-detection-dataset')
COMPLETE_DATA_TRAIN = os.path.join(DATA_DIR, 'complete_train.csv')
VALUATION_FILE = os.path.join(DATA_DIR, 'test.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
SUMMARY_DICT = os.path.join(BASE_DIR, 'summary')
WEIGHTS = os.path.join(BASE_DIR, 'vgg16.npy')
CHECKPOINT = 'checkpoint/'  # Relatve to root folder, not need absolute path

# Config for training
IMAGE_SIZE = 128        # Image size for input training
BATCH_SIZE = 42         # How much image will train in a batch
TEST_SIZE = 0.20        # Composition train and test data
EPOCH_NUM = 50          # Total epoch for training
SKIP_STEP = 20          # How much step will pass for evaluation
ITERATE_PER_EPOCH = 10  # Iteration per epoch
NUM_ITERATE_TEST = 5    # Evaluation range
CLASS_NUM = 42          # Total class
IMAGE_CHANNEL = 3       # Image channel
LEARNING_RATE = 0.001   # Learning rate
MAX_TO_KEP = 5          # How much checkpoint will be keep

