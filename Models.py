import tensorflow as tf
import torch
import numpy as np

from tensorflow.python.keras.models import Model

class TensorFlowApp(Model):
    def __init__(self):
        super(TensorFlowApp, self).__init__()
        self.CNN = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(100, 2)),
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),  # 100 packets, 2 features
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),  # 100 packets, 2 features
        ])
        self.clf = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(8, activation='softmax')
        ])

    def call(self, x):
        x = self.CNN(x)
        x = self.clf(x)
        return x


class TensorFlowMalware(Model):
    def __init__(self):
        super(TensorFlowMalware, self).__init__()
        self.CNN = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(100, 2), padding='same'),
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
    def call(self, x):
        return self.CNN(x)

class PyTorchApp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = torch.nn.Sequential(torch.nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool1d(kernel_size=3),
                                       torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool1d(kernel_size=3),
                                       torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
                                       torch.nn.ReLU())
        self.clf = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_features=256, out_features=128),
                                       torch.nn.Linear(in_features=128, out_features=8))

    def forward(self, x):
        x = self.CNN(x)
        return self.clf(x)

class PyTorchMalware(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = torch.nn.Sequential(torch.nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool1d(kernel_size=3),
                                       torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool1d(kernel_size=3),
                                       torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
                                       torch.nn.ReLU())
        self.clf = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_features=256, out_features=128),
                                       torch.nn.Linear(in_features=128, out_features=2))
    def forward(self, x):
        x = self.CNN(x)
        return self.clf(x)


class K1App(Model):
    def __init__(self):
        super(K1App, self).__init__()
        self.CNN = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(100, 2)),
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),  # 100 packets, 2 features
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),  # 100 packets, 2 features
        ])
        self.clf = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(9, activation='softmax')
        ])

    def call(self, x):
        x = self.CNN(x)
        x = self.clf(x)
        return x


class K1Malware(Model):
    def __init__(self):
        super(K1Malware, self).__init__()
        self.CNN = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(100, 2), padding='same'),
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool1D(pool_size=3, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
    def call(self, x):
        return self.CNN(x)