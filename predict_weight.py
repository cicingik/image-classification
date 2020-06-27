# -*- coding: utf-8 -*-
import json
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cnf.config import IMAGE_SIZE, MODEL, TEST_DIR, TEST_FILE, IMAGE_CHANNEL
from cnf.argument import options
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# argparser = argparse.ArgumentParser(description=__doc__)
# argparser.add_argument(
#     '-d', '--directory',
#     type=str,
#     default='None',
#     help='Folder image that will predict')
# args = argparser.parse_args()

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


def predict(filename, model):
    im = load_image(filename, show=False)
    pre = model.predict(im)
    img_class = np.argmax(pre)
    return img_class

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception, InceptionResNetV2

def main(model_type, filemodel):
    model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))
    for layer in model.layers[:10]:
        layer.trainable = False
    top_model = Sequential()
    top_model.add(model)
    top_model.add(GlobalAveragePooling2D())
    top_model.add(Flatten())
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(42, activation='softmax'))

    df = pd.read_csv(TEST_FILE, delimiter=',')
    df['file_path'] = df.apply(lambda x: os.path.join(TEST_DIR, x.filename), axis=1)
    df['category'] = df.apply(lambda x: predict(x.file_path, top_model), axis=1)
    dk = df[['filename', 'category']]
    dk.to_csv(f'{model_type}-predict.csv', mode='a', header=True, index=False, sep=',')
    print(dk)


if __name__ == '__main__':
    # main()
    # df = pd.read_csv('predict.csv')
    # dk = df[['filename', 'class']]
    # dk.rename(columns={
    #     'class': 'category'
    # }, inplace=True)
    # dk.to_csv('submit.csv', mode='a', header=True, index=False, sep=',')
    # print(dk)
    p = options()
    args = p.parse_args()

    if not all([args.model, args.filemodel]):
        p.print_help()
        sys.exit(2)

    main(args.model, args.filemodel)