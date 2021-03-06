# -*- coding: utf-8 -*-
import os
import sys
import colorama
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from cnf.config import (IMAGE_SIZE, IMAGE_CHANNEL, BATCH_SIZE, EPOCH_NUM,
                        MODEL_DIR, TRAIN_VERSION, LEARNING_RATE, DECAY)


class Models:
    def __init__(self, training_set,  validation_set, model_type: str):
        self.model_type = model_type
        self.train_set = training_set
        self.validation_set = validation_set
        self.model = self.build_model()

    @property
    def __build_vgg16(self):
        model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))

        for layer in model.layers[:-5]:
            layer.trainable = False
        top_model = Sequential()
        top_model.add(model)
        top_model.add(Flatten())
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))

        return top_model

    def build_model(self):
        if self.model_type == 'vgg16':
            model = self.__build_vgg16
        else:
            raise Exception('Invalid model name')

        return model

    @property
    def train(self):
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=LEARNING_RATE, decay=DECAY),
            metrics=['accuracy'])
        self.model.fit_generator(
            self.train_set,
            steps_per_epoch=self.train_set.n // BATCH_SIZE,
            epochs=EPOCH_NUM,
            validation_data=self.validation_set,
            validation_steps=self.validation_set.n // BATCH_SIZE)

        self.model.save(f'{MODEL_DIR}/{TRAIN_VERSION}.h5', save_format='h5')
        self.model.save(f'{MODEL_DIR}/{TRAIN_VERSION}.pb', save_format='tf')

        return
