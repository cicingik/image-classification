# -*- coding: utf-8 -*-
from data_loader.data_loader import DataLoader
from models.models import Models
from cnf.argument import options
import colorama
import sys
import os


def main(model):
    data = DataLoader()
    model = Models(data.train_set, data.valuation_set, model_type=model)
    model.train()


if __name__ == '__main__':
    p = options()
    args = p.parse_args()

    if not all([args.model]):
        p.print_help()
        sys.exit(2)

    main(args.model)
