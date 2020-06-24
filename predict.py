# -*- coding: utf-8 -*-
import cv2
import json
import requests
import argparse
import numpy as np
from bunch import Bunch
from cnf.config import CLASSIFICATION_URL

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '-i', '--image',
    type=str,
    default='None',
    help='File image that will predict')
args = argparser.parse_args()


img = cv2.imread(args.image)
image = cv2.resize(img, (128, 128))
payload = {"instances": [{'images': image.tolist()}]}
r = requests.post(CLASSIFICATION_URL, json=payload)
b = r.content.decode('utf8').replace("'", '"')
data = json.loads(b)
config = Bunch(data)
img_class = np.argmax(config.predictions)
print(f"Result prediction image: {img_class}")
