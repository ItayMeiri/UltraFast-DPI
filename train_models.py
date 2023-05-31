# tensorflow models

# Application
import tensorflow as tf
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from Models import TensorFlowApp
import my_utils
import torch
import torch.optim as optim
import numpy as np


####### TENSORFLOW MODELS #######
def tensorflow_app():
    print("Tensorflow App")
    data, labels = my_utils.load_known_applications()
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20, random_state=42)

    fx = TensorFlowApp()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    fx.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    fx.fit(train_data, train_labels, epochs=15, validation_data=(test_data, test_labels), batch_size=32,
           callbacks=[callback])

    res = fx.evaluate(test_data, test_labels)
    print(res)

    fx.save("TensorFlowApp")


# Malware
def tensorflow_malware():
    print("Tensorflow Malware")
    from Models import TensorFlowMalware
    data, labels = my_utils.load_known_malware()
    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20, random_state=42)

    fx = TensorFlowMalware()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    fx.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    fx.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels), batch_size=32,
           callbacks=[callback])

    res = fx.evaluate(test_data, test_labels)
    print(res)
    fx.save("TensorFlowMalware")


####### ####### #######


####### PYTORCH MODELS #######

# Application

def pytorch_app():
    from Models import PyTorchApp

    model = PyTorchApp()

    print("PyTorch App")
    data, labels = my_utils.load_known_applications()
    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20, random_state=42)


    Xtrain = torch.from_numpy(train_data)
    Ytrain = torch.from_numpy(np.array(train_labels))

    Ytest = torch.from_numpy(np.array(test_labels))
    Ytrain = Ytrain.type(torch.LongTensor)

    Xtest = torch.from_numpy(test_data)
    Ytest = Ytest.type(torch.LongTensor)

    # Using the in built DataLoader
    my_dataset = TensorDataset(Xtrain, Ytrain)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        # loop over the dataset multiple times
        print("epoch: ", epoch + 1)
        print_step = 2000
        running_loss = 0.0
        for i, data in enumerate(my_dataloader):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    PATH = 'models/pytorch/PyTorchApp.pth'
    torch.save(model, PATH)


# Malware

def pytorch_malware():
    print("PyTorch Malware")
    from Models import PyTorchMalware
    model = PyTorchMalware()

    data, labels = my_utils.load_known_malware()
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20, random_state=42)

    Xtrain = torch.from_numpy(train_data)
    Ytrain = torch.from_numpy(np.array(train_labels))

    Ytest = torch.from_numpy(np.array(test_labels))
    Ytrain = Ytrain.type(torch.LongTensor)

    Xtest = torch.from_numpy(test_data)
    Ytest = Ytest.type(torch.LongTensor)

    # Using the in built DataLoader
    my_dataset = TensorDataset(Xtrain, Ytrain)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        # loop over the dataset multiple times
        print("epoch: ", epoch + 1)
        running_loss = 0.0
        for i, data in enumerate(my_dataloader):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    PATH = 'models/pytorch/PyTorchMalware.pth'
    torch.save(model, PATH)


####### ####### #######


# K+1(also tensorflow) ######

def k1_app():
    from Models import K1App

    print("K1App")

    data, labels = my_utils.load_known_applications()
    data_u, labels_u = my_utils.load_unknown_applications(training=True)
    # insert training from both sets
    data, _, labels, _ = train_test_split(data, labels, test_size=0.20, random_state=42)
    # data_u, _, labels_u, _ = train_test_split(data_u, labels_u, test_size=0.5, random_state=42) # 0.5 train 0.5 test

    labels_u[labels_u > -1] = np.max(labels) + 1
    train_data = np.concatenate([data, data_u])
    train_labels = np.concatenate([labels, labels_u])

    fx = K1App()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    fx.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    fx.fit(train_data, train_labels, epochs=15, batch_size=32,
           callbacks=[callback])

    fx.save("K1App")

def k1_malware():

    # Malware
    print("K1Malware")

    from Models import K1Malware

    data, labels = my_utils.load_known_malware()
    data_u, labels_u = my_utils.load_unknown_malware(K1=True, training=True)

    # split unknown data/labels
    data, _, labels, _ = train_test_split(data, labels, test_size=0.20, random_state=42)
    data_u, _, labels_u, _ = train_test_split(data_u, labels_u, test_size=0.5, random_state=42) # 0.5 train 0.5 test



    # # insert training from both sets
    # data, _, labels, _ = train_test_split(data, labels, test_size=0.25, random_state=42)
    # data_u, _, labels_u, _ = train_test_split(data_u, labels_u, test_size=0.25, random_state=42)

    labels_u[labels_u > -1] = np.max(labels) + 1
    train_data = np.concatenate([data, data_u])
    train_labels = np.concatenate([labels, labels_u])

    train_data = train_data.reshape(train_data.shape[0], train_data.shape[2], train_data.shape[1])
    fx = K1Malware()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    fx.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    fx.fit(train_data, train_labels, epochs=15, batch_size=32,
           callbacks=[callback])

    # res = fx.evaluate(test_data, test_labels)
    # print(res)

    fx.save("K1Malware")


# tensorflow_app()
tensorflow_malware()
#
pytorch_app()
pytorch_malware()

k1_app()
k1_malware()
