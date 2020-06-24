# -*- coding: utf-8 -*-
from data_loader.data_loader import DataLoader
from models.vgg16 import Models


data = DataLoader()
model = Models(data.train_set, data.len_train_set, data.valuation_set, data.len_valuation_set)
model.train()
