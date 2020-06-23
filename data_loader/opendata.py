# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
from cnf.config import DATA_DIR, TRAIN_FILE, VALUATION_FILE

log = logging.getLogger(__name__)


def full_path(data_type: str, category: int, filename: str):
    category = '0'+str(category) if len(str(category)) < 2 else str(category)
    file_path = os.path.join(DATA_DIR, f'{data_type}/{data_type}/{category}/{filename}')
    return file_path


def open_data_train():
    df = pd.read_csv(TRAIN_FILE, delimiter=',')
    df['file_path'] = df.apply(lambda row: full_path('train', row.category, row.filename), axis=1)
    return df


def open_data_test():
    df = pd.read_csv(VALUATION_FILE, delimiter=',')
    df['file_path'] = df.apply(lambda row: full_path('test', row.category, row.filename), axis=1)
    return df


def generate_complete_data(dest_folder: str):
    df = open_data_train()
    COMPLETE_DATA_TRAIN = os.path.join(dest_folder, 'complete_train.csv')
    df.to_csv(COMPLETE_DATA_TRAIN, mode='a', header=True, index=False, sep=',')
    print(df)
    log.info(f"Save complete data train with name {COMPLETE_DATA_TRAIN}")
