# -*- coding: utf-8 -*-
from data_loader.data_loader import DataLoader
from models.models import Models
from cnf.argument import options
import sys


def image_size(model_type):
    if model_type == 'vgg16':
        return 224
    elif model_type == 'xception':
        return 299
    elif model_type == 'inceptionresnetv2':
        return 299
    else:
        raise Exception('Invalid model name')

def main(model):
    data = DataLoader(image_size(model))
    model = Models(data.train_set, data.valuation_set, model, image_size(model))
    model.train


if __name__ == '__main__':
    p = options()
    args = p.parse_args()

    if not all([args.model]):
        p.print_help()
        sys.exit(2)

    main(args.model)
