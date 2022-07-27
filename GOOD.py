# general out of distribution detector
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


class GOOD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clf = None
        self.model = None
        self.bp_layer = ""
        self.threshold = 0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.grads = None

    def score(self, x, y, weights=(0.5, 0.5)):
        w1, w2 = weights
        return w1 * x + w2 * y

    # requires OOD samples to be labled OOD.
    def calibrate(self, X, y, max_rejected=0.7, first_run=False):
        threshold = 10.0
        max_runs = 100

        mx_t = -1
        mx_score = -1
        improvement = False
        mx = torch.max(y)
        prev_score = -1
        length = y.size

        if first_run:
            r = self(X)
        for i in range(max_runs):
            # num of OOD detected
            detected_ood = np.greater_equal(self.grads, threshold)

            # true positives
            tp = torch.sum(y[detected_ood] >= mx)

            # false positives
            fp = torch.sum(y[~detected_ood] < mx)

            score = self.score(tp, fp)

            if score > mx_score:
                mx_score = score
                mx_t = threshold

            if score >= prev_score:
                prev_score = score
                threshold = threshold / 2
            else:
                prev_score = score
                threshold = threshold * 1.9

        return mx_t

    def set_clf(self, clf):
        self.clf = clf

    def set_model(self, model, last_layer):

        if not isinstance(last_layer, str):
            self.model = model
            self.bp_layer = last_layer
            return

        # possibly just passing the last layer itself is better, not tested yet
        # self.model = model
        # self.bp_layer = self.model.__getattr__(last_layer)
        # self.bp_layer = self.bp_layer[len(self.bp_layer) - 1]

        # This might be a more general way:
        self.model = model
        layers = self.model.__dict__['_modules']  # returns all the modules
        layers = layers[last_layer]  # gets the last layer name, for example a classifier with dense/linear layers.
        if hasattr(layers, "__len__"):
            self.bp_layer = layers[-1]  # last
        else:
            self.bp_layer = layers

    def set_thresh(self, t):
        self.threshold = t

    def set_criterion(self, criterion):
        self.criterion = criterion

    def forward(self, x):
        # Model outputs - we'll be padding them later
        o = self.model(x)

        # calculate the "ground truths", this will be used to calculate the BP step
        raw, ground_truths = torch.max(o, axis=1)

        # probably the fastest way right now
        unknown_dataset = TensorDataset(x, ground_truths)
        unknown_dataloader = DataLoader(unknown_dataset, batch_size=1, shuffle=False)  # batch size must be 1 for now

        grads = np.zeros((o.shape[0]))
        for i, data in enumerate(unknown_dataloader):
            # gets the input
            inputs, labels = data

            # possible -
            self.model.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            # BP through the whole network - this part can be more efficient by doing it on just the last layer
            loss.backward()

            # get the gradient
            grad = self.bp_layer.bias.grad

            # L2 Norm - calculate magnitude of the gradient
            grads[i] = torch.sqrt(torch.sum(torch.square(grad)))
        self.grads = grads
        r = F.pad(input=o, pad=(0, 1))  # adds a column of 0s
        r[:, -1][np.greater_equal(grads, self.threshold)] = 999  # r[:, -1] are all of the padded columns, we're
        # setting them to 999 in the indexes where grad >= threshold
        return r
