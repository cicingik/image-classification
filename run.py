# -*- coding: utf-8 -*-
from data_loader.data_loader import DataLoader
from models.models import Models
from cnf.argument import options
import sys


def image_size(model_type):
    if model_type == 'vgg16':
        return 224
    elif model_type == 'xception' or model_type == 'inceptionresnetv2' or model_type == 'efficientnetb6' or model_type == 'efficientnetb7' or model_type == 'efficientnetl2':
        return 299
    else:
        raise Exception('Invalid model name')

def main(model=None, savedmodel=None):
    if model is not None:
        data = DataLoader(image_size(model))
        model = Models(data.train_set, data.valuation_set, model, image_size(model))
        model.train
    elif savedmodel is not None:
        savedmodel.train


if __name__ == '__main__':
    p = options()
    args = p.parse_args()

    if args.model is not None and args.savedmodel is None:
        main(model=args.model)
    elif args.model is None and args.savedmodel is not None:
        main(savedmodel=args.savedmodel)
    else:
        p.print_help()
        sys.exit(2)
