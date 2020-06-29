# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnf.config import BATCH_SIZE, TRAIN_FILE, TEST_FILE, TRAIN_DIR, TEST_SIZE, TEST_DIR


class DataLoader:

    def __init__(self, image_size):
        self.image_size = image_size
        self.df_data_train = self.__open_train_data
        self.df_data_test = self.__open_test_data
        self.datagen_train = self.__generate_data
        self.train_set = self.__trains_set
        self.len_train_set = self.__trains_set.n
        self.valuation_set = self.__valuation_set
        self.len_valuation_set = self.__valuation_set.n
        self.test_set = self.__test_set

    @property
    def __open_train_data(self):
        data = pd.read_csv(TRAIN_FILE, delimiter=',')
        data['category'] = data.apply(lambda row:
                                      '0' + str(row.category) if len(str(row.category)) < 2 else str(row.category),
                                      axis=1)
        data['filename'] = data.apply(lambda row: '/'.join([row.category, row.filename]), axis=1)
        data = shuffle(data)
        return data

    @property
    def __open_test_data(self):
        data = pd.read_csv(TEST_FILE, delimiter=',')
        data['category'] = data.apply(lambda row:
                                      '0' + str(row.category) if len(str(row.category)) < 2 else str(row.category),
                                      axis=1)
        data = shuffle(data)
        return data

    # Please provide some augmentation image here for reduce overfitting
    @property
    def __generate_data(self):
        datagen = ImageDataGenerator(rescale=1. / 255.,
                                     vertical_flip=True,
                                     horizontal_flip=True,
                                     rotation_range=90,
                                     shear_range=0.5,
                                     zoom_range=0.5,
                                     validation_split=TEST_SIZE)
        return datagen

    @property
    def __trains_set(self):
        training_set = self.datagen_train.flow_from_dataframe(
            dataframe=self.df_data_train, directory=TRAIN_DIR,
            x_col='filename', y_col='category', subset='training',
            batch_size=BATCH_SIZE, seed=42, shuffle=True,
            class_mode='sparse', target_size=(self.image_size, self.image_size))
        return training_set

    @property
    def __valuation_set(self):
        val_set = self.datagen_train.flow_from_dataframe(
            dataframe=self.df_data_train, directory=TRAIN_DIR,
            x_col='filename', y_col='category', subset='validation',
            batch_size=BATCH_SIZE, seed=42, shuffle=True,
            class_mode='sparse', target_size=(self.image_size, self.image_size))
        return val_set

    @property
    def __test_set(self):
        datagen = ImageDataGenerator(rescale=1. / 255.)
        test_generator = datagen.flow_from_dataframe(
            dataframe=self.df_data_test,
            directory=TEST_DIR,
            x_col="filename",
            y_col=None,
            batch_size=BATCH_SIZE,
            seed=42,
            shuffle=False,
            class_mode=None,
            target_size=(self.image_size, self.image_size))
        return test_generator

    def __dict__(self):
        return {
            'df_data_train': self.df_data_train,
            'df_data_test': self.df_data_test,
            'datagen_train': self.datagen_train,
            'data_train': self.train_set,
            'len_data_train': self.len_train_set,
            'data_valuation': self.valuation_set,
            'len_data_valuation': self.len_valuation_set,
            'data_test': self.test_set,
            'image_size': self.image_size
        }
