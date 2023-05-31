import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F

import my_utils


# returns the gradient magnitudes.
# gradients > threshold -> OOD

def shadow_backpropagation(model, bp_layer, x):
    # Model outputs - we'll be padding them later
    criterion = torch.nn.CrossEntropyLoss()
    x.requires_grad = True
    o = model(x)  # [0.1, 0.02, 0.5..]

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
        model.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        # BP through the whole network - this part can be more efficient by doing it on just the last layer
        loss.backward()

        # get the gradient
        grad = bp_layer.weight.grad

        # L2 Norm - calculate magnitude of the gradient
        grads[i] = torch.sqrt(torch.sum(torch.pow(grad, 2)))
    grads = grads
    r = F.pad(input=F.softmax(o, dim=1), pad=(0, 1))  # adds a column of 0s

    # r[:, -1][np.greater_equal(grads, threshold)] = 999  # r[:, -1] are all of the padded columns, we're
    # # setting them to 999 in the indexes where grad >= threshold
    return grads, r

# returns the scores for a certain epsilon
# scores > threshold -> OOD
def ODIN(model, data,eps):
    perturbed_data = my_utils.create_perturbed_ds(data, model, eps)
    scores = np.max(model.predict(perturbed_data), axis=1)
    return scores

# predicting last class -> OOD
def kplus1(model, data):
    return np.argmax(model.predict(data), axis=1)


