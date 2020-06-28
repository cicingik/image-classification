# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import Xception, InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from cnf.config import (IMAGE_SIZE, IMAGE_CHANNEL, BATCH_SIZE, EPOCH_NUM,
                        LEARNING_RATE, DECAY, CLASS_NUM)


def add_fc_layer(model):
    for layer in model.layers[:len(model.layers) // 2]:
        layer.trainable = False

    top_model = Sequential()
    top_model.add(model)
    top_model.add(GlobalAveragePooling2D())
    top_model.add(Flatten())
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(CLASS_NUM, activation='softmax'))

    return top_model


class Models:
    def __init__(self, training_set,  validation_set, model_type: str, image_size):
        self.model_type = model_type
        self.image_size = image_size
        self.train_set = training_set
        self.validation_set = validation_set
        self.model = self.build_model()

    @property
    def __build_vgg16(self):
        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(vgg16_model)

    @property
    def __build_xception(self):
        xception_model = Xception(weights='imagenet', include_top=False, input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(xception_model)

    @property
    def __build_inceptionresnetv2(self):
        inceptionresnetv2_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(self.image_size, self.image_size, IMAGE_CHANNEL))
        return add_fc_layer(inceptionresnetv2_model)

    def build_model(self):
        if self.model_type == 'vgg16':
            model = self.__build_vgg16
        elif self.model_type == 'xception':
            model = self.__build_xception
        elif self.model_type == 'inceptionresnetv2':
            model = self.__build_inceptionresnetv2
        else:
            raise Exception('Invalid model name')

        return model

    @property
    def train(self):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizers.Adam(lr=LEARNING_RATE),
            metrics=['accuracy'])

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
        model_checkpoint_callback = ModelCheckpoint(
            filepath='/content/drive/My Drive/sopikodelig/{epoch:02d}-{val_accuracy:.4f}.hdf5',
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        self.model.fit(
            x = self.train_set,
            steps_per_epoch=self.train_set.n // BATCH_SIZE,
            epochs=EPOCH_NUM,
            workers=6,
            callbacks=[model_checkpoint_callback],
            validation_data=self.validation_set,
            validation_steps=self.validation_set.n // BATCH_SIZE)

        # self.model.save(f'{MODEL_DIR}/{TRAIN_VERSION}.h5', save_format='h5')
        # self.model.save(f'{MODEL_DIR}/{TRAIN_VERSION}.pb', save_format='tf')

        return
