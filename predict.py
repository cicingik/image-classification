# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cnf.config import IMAGE_SIZE, TEST_DIR, TEST_FILE
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


def predict(filename: str, model) -> int:
    im = load_image(filename, show=False)
    pre = model.predict(im)
    img_class = np.argmax(pre)
    return img_class


def main(savedmodel):
    model = load_model(savedmodel)
    df = pd.read_csv(TEST_FILE, delimiter=',')
    df['file_path'] = df.apply(lambda x: os.path.join(TEST_DIR, x.filename), axis=1)
    df['category'] = df.apply(lambda x: predict(x.file_path, model), axis=1)
    dk = df[['filename', 'category']]
    dk.to_csv(f'{savedmodel}-predict.csv', mode='a', header=True, index=False, sep=',')
    print(dk)


if __name__ == '__main__':
    p = options()
    args = p.parse_args()

    if not all([args.savedmodel]):
        p.print_help()
        sys.exit(2)

    main(args.savedmodel)