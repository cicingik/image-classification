# -*- coding: utf-8 -*-
import json
import os
import requests
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cnf.config import IMAGE_SIZE, MODEL, TEST_DIR, TEST_FILE
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# argparser = argparse.ArgumentParser(description=__doc__)
# argparser.add_argument(
#     '-d', '--directory',
#     type=str,
#     default='None',
#     help='Folder image that will predict')
# args = argparser.parse_args()

model = load_model(MODEL)


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


def predict(filename: str) -> int:
    print(f'')
    im = load_image(filename, show=False)
    pre = model.predict(im)
    img_class = np.argmax(pre)
    return img_class


def main():
    df = pd.read_csv(TEST_FILE, delimiter=',')
    df['file_path'] = df.apply(lambda x: os.path.join(TEST_DIR, x.filename), axis=1)
    df['category'] = df.apply(lambda x: predict(x.file_path), axis=1)
    dk = df[['filename', 'category']]
    dk.to_csv('predict.csv', mode='a', header=True, index=False, sep=',')
    print(dk)


if __name__ == '__main__':
    main()
    # df = pd.read_csv('predict.csv')
    # dk = df[['filename', 'class']]
    # dk.rename(columns={
    #     'class': 'category'
    # }, inplace=True)
    # dk.to_csv('submit.csv', mode='a', header=True, index=False, sep=',')
    # print(dk)