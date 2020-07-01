# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import Xception, InceptionResNetV2
from efficientnet.tfkeras import EfficientNetB6, EfficientNetB7, EfficientNetL2
from tensorflow.keras.applications.vgg16 import VGG16
from cnf.config import (IMAGE_CHANNEL, BATCH_SIZE, EPOCH_NUM,
                        LEARNING_RATE, DECAY, CLASS_NUM, MODEL_DIR)
import datetime


def add_fc_layer(model):
    top_model = Sequential()
    top_model.add(model)
    top_model.add(Dense(CLASS_NUM, activation='softmax'))

    return top_model


class Models:
    def __init__(self, training_set, validation_set, model_type, image_size):
        self.model_type = model_type
        self.image_size = image_size
        self.train_set = training_set
        self.validation_set = validation_set
        self.model = self.build_model()

    @property
    def __build_vgg16(self):
        vgg16_model = VGG16(weights='imagenet', include_top=False,
                            input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(vgg16_model)

    @property
    def __build_xception(self):
        xception_model = Xception(weights='imagenet', include_top=False, pooling='avg',
                                  input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(xception_model)

    @property
    def __build_inceptionresnetv2(self):
        inceptionresnetv2_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg',
                                                    input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(inceptionresnetv2_model)

    @property
    def __build_efficientnetb6(self):
        efficientnetb6_model = EfficientNetB6(weights='imagenet', include_top=False, pooling='avg',
                                              input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(efficientnetb6_model)

    @property
    def __build_efficientnetb7(self):
        efficientnetb7_model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg',
                                                    input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(efficientnetb7_model)

    @property
    def __build_efficientnetl2(self):
        efficientnetl2_model = EfficientNetL2(weights='imagenet', include_top=False, pooling='avg',
                                              input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(efficientnetl2_model)

    def build_model(self):
        if self.model_type == 'vgg16':
            model = self.__build_vgg16
        elif self.model_type == 'xception':
            model = self.__build_xception
        elif self.model_type == 'inceptionresnetv2':
            model = self.__build_inceptionresnetv2
        elif self.model_type == 'efficientnetb6':
            model = self.__build_efficientnetb6
        elif self.model_type == 'efficientnetb7':
            model = self.__build_efficientnetb7
        elif self.model_type == 'efficientnetl2':
            model = self.__build_efficientnetl2
        else:
            raise Exception('Invalid model name')

        return model

    @property
    def train(self):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizers.Adam(lr=LEARNING_RATE),
            metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        model_checkpoint_callback = ModelCheckpoint(
            filepath='save_models/{epoch:02d}-{val_accuracy:.4f}.hdf5',
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        self.model.fit(
            x=self.train_set,
            # steps_per_epoch=self.train_set.n // BATCH_SIZE,
            epochs=EPOCH_NUM,
            workers=1,
            callbacks=[model_checkpoint_callback, reduce_lr, tensorboard_callback],
            validation_data=self.validation_set)
            # validation_steps=self.validation_set.n // BATCH_SIZE)

        return
