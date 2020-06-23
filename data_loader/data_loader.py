import cv2
import numpy as np
import pandas as pd
from cnf.config import IMAGE_SIZE, BATCH_SIZE, TEST_SIZE
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, complete_data_train: str):
        self.complete_data_train = complete_data_train
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_dataset()
        self.len_train = len(self.X_train)
        self.len_test = len(self.X_test)
        self.current_index = 0

    @staticmethod
    def open_image(file: str):
        img = cv2.imread(file)
        if img is None:
            print(f"Image can not read. Details file: {file}")
            pass

        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img

    def get_dataset(self):
        dg = pd.read_csv(self.complete_data_train, delimiter=',')
        X = dg['file_path'].tolist()
        y = dg['category'].tolist()
        return train_test_split(X, y, test_size=TEST_SIZE)

    def slice_data(self, X, y, batch_slice):
        image_batch = [self.open_image(X[i]) for i in batch_slice]
        label_batch = list(map(lambda p: y[p], batch_slice))
        return image_batch, label_batch

    def get_batch(self, trainable=True):
        X = self.X_train if trainable else self.X_test
        y = self.y_train if trainable else self.y_test

        num_files = self.len_train if trainable else self.len_test
        end_index = self.current_index + BATCH_SIZE

        batch_slice = np.random.choice(num_files, BATCH_SIZE)

        image_batch, label_batch = self.slice_data(X, y, batch_slice)
        image_batch = np.reshape(np.squeeze(np.stack([image_batch])), newshape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
        label_batch = np.stack(label_batch)

        self.current_index = end_index

        if self.current_index > num_files:
            self.current_index = self.current_index - num_files

        return image_batch, label_batch
