import itertools

import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import json

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, \
    classification_report, precision_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from tensorflow import keras


def parse_normalize(data):
    data = np.array(data.apply(lambda cell: np.array(json.loads(cell)).astype(np.float32)).to_list())
    # data = np.array(data.apply(lambda cell: np.array(json.loads(cell)).astype(np.float32)).to_list())
    return tf.keras.utils.normalize(data)


def load_known_malware(benign=True):
    df = pd.read_csv("datasets/MTAB_dataset.csv", low_memory=False)
    if not benign:
        df = df.drop(df[df["label"] == "benign"].index)
    data_u = df["udps.n_bytes_per_packet"]
    data_u = parse_normalize(data_u)
    labels = df["label"]
    labels = pd.Series(np.where(labels.values == 'benign', 0, 1), labels.index)
    return data_u, labels

items = ["emotet", "dridex", "benign"]
df = pd.read_csv("datasets/USTCB_dataset.csv", low_memory=False)
for item in items:
    df = df.drop(df[df["malware_family"] == item].index)


# excluding benign and shared malwares known dataset
def load_unknown_malware(K1=False, training=False):
    items = ["emotet", "dridex", "benign"]
    df = pd.read_csv("datasets/USTCB_dataset.csv", low_memory=False)
    if K1 is False:
        for item in items:
            df = df.drop(df[df["malware_family"] == item].index)
    else:
        if training:
            items.append("shifu")
            items.append("zeus")
            items.append("virut")
        else:
            items.append("htbot")
            items.append("miuref")
            items.append("neris")
            items.append("nsis")

    for item in items:
        df = df.drop(df[df["malware_family"] == item].index)
    data_u = df["udps.n_bytes_per_packet"]
    data_u = parse_normalize(data_u)
    labels = df["label"]
    # change labels to 1s or 0s
    labels = pd.Series(np.where(labels.values == 'benign', 0, 1), labels.index)

    return data_u, labels


def load_known_applications():
    df = pd.read_csv("datasets/known_apps.csv", names=["data", "labels"], header=1)
    data = df["data"]
    labels = df["labels"]
    data = parse_normalize(data)

    return data, labels


def load_unknown_applications(K1=False, training=False):
    df = pd.read_csv("datasets/unknown_apps.csv", names=["data", "labels"], header=1)
    data = df["data"]
    labels = df["labels"]
    data = parse_normalize(data)

    # removing labels 0, 4 for testing
    indx = np.logical_or(labels == 0, labels == 4)

    # if K1:
    if training:
        data = data[indx]
        labels = labels[indx]
    else:
        data = data[~indx]
        labels = labels[~indx]
    return data, labels


def get_malware_model(platform="tensorflow", k_plus_one=False):
    if platform.lower() == "tensorflow":
        if k_plus_one:
            return keras.models.load_model("models/tensorflow/K1Malware")
        return keras.models.load_model("models/tensorflow/TensorFlowMalware")
    elif platform.lower() == "pytorch":
        from Models import PyTorchMalware
        return torch.load('models/pytorch/PyTorchMalware.pth')
    else:
        return "Invalid name"


def get_application_model(platform="tensorflow", k_plus_one=False):
    if platform.lower() == "tensorflow":
        if k_plus_one:
            return keras.models.load_model("models/tensorflow/K1App")
        return keras.models.load_model("models/tensorflow/TensorFlowApp")
    elif platform.lower() == "pytorch":
        from Models import PyTorchApp
        return torch.load('models/pytorch/PyTorchApp.pth')
    else:
        return "Invalid name"


def test_train_split(data, labels):
    pass


# to do - vectorize this - it's really inefficient.

def perturb_input(input, model, epsilon, show_perturbation=False):
    input_tensor = tf.convert_to_tensor(input)

    # records operations for automatic differentiation
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        outputs = model(input_tensor, training=False)
        loss = tf.reduce_max(outputs, axis=1)
        loss = tf.reduce_mean(loss)

    # gradient calculation
    gradients = tape.gradient(loss, input_tensor)

    # returns the truth value ( x >= y ) element-wise
    gradients = tf.math.greater_equal(gradients, 0)

    # fixes some compatibility issues
    gradients = tf.cast(gradients, tf.double)

    gradients = np.sign(gradients)

    # return the perturbed input
    if show_perturbation:
        show_image_list([input_tensor, epsilon * gradients, input_tensor - epsilon * gradients])
    return input_tensor - epsilon * gradients


def create_perturbed_ds(inputs, model, eps):
    # inputs = inputs[:(inputs.shape[0]//10)]
    perturbed_inputs = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
    for i, X in enumerate(inputs):
        perturbed_inputs[i] = perturb_input(tf.expand_dims(X, 0), model, epsilon=eps)
        if i % (perturbed_inputs.shape[0] / 7) == 0:
            print(np.around(i / inputs.shape[0], 2) * 100, "%")

    print("Finished creating perturbed dataset")
    return perturbed_inputs


def evaluate_without_model(data, labels, show_matrix=False, excluded=False, samples=False, use_torch=True, display_labels = None, ood=True, title="title"):
    if use_torch:
        preds = data
        mx_label = torch.max(labels)
    else:
        preds = data
        mx_label = np.max(labels)
    if not ood:
        mx_label+=1
    rejected_percent = np.count_nonzero(preds == mx_label) / data.shape[0]
    # print("correctly out: ", np.count_nonzero(labels[preds == mx_label] == mx_label), "out of: ",
    #       np.count_nonzero(labels == mx_label))
    # print("correctly in: ", np.count_nonzero(labels[preds != mx_label] != mx_label), "out of: ",
    #       np.count_nonzero(labels != mx_label))

    true_positives, true_negatives = 0, 0
    if ood:
        true_positives = np.count_nonzero(labels[preds == mx_label] == mx_label) / np.count_nonzero(labels == mx_label)
        true_negatives = np.count_nonzero(labels[preds != mx_label] != mx_label) / np.count_nonzero(labels != mx_label)

    excluded_preds = preds
    excluded_labels = labels
    if excluded:
        exclude = preds == mx_label
        excluded_preds = preds[~exclude]
        excluded_labels = labels[~exclude]


    if use_torch:
        labels = labels.detach().numpy()
        preds = preds.detach().numpy()
        excluded_labels = excluded_labels.detach().numpy()
        excluded_preds = excluded_preds.detach().numpy()

    KP = np.sum(labels[labels!=mx_label] == preds[labels!= mx_label])
    UP = np.sum(labels[labels == mx_label] == preds[labels == mx_label])
    KU = np.sum(preds[labels!=mx_label] == mx_label) # Known class, classified as unknown
    KN = np.sum(labels[labels!=mx_label] != preds[labels != mx_label]) # Known class + incorrect prediction
    UN = np.sum(preds[labels==mx_label] != mx_label) # Unknown class, wrongly identified as known

    purity_rate = (KP + UP) / (KP + KN + KU + UP + UN)
    purity_rate2 = KP / (KP + KN + KU)




    acc = accuracy_score(excluded_labels, excluded_preds)
    recall = recall_score(excluded_labels, excluded_preds, average="macro", zero_division=0)
    precision = precision_score(excluded_labels, excluded_preds,average="macro", zero_division=0)
    f1 = f1_score(excluded_labels, excluded_preds, average="macro", zero_division=0)

    if show_matrix:
        plt.rcParams.update({'font.size': 18})
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams["figure.figsize"] = [16, 9]
        cm = confusion_matrix(labels, preds)
        df = pd.DataFrame(cm)
        df.columns = display_labels
        df.index = display_labels

        df.to_csv("images/" + title + ".csv")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        print("******")
        print(title)
        print(cm)

        print("acc:", acc)
        print("recall:", recall)
        print("precision:", precision)
        print("F1:", f1)

        print("TNR:", true_negatives, "\nTPR:", true_positives)
        print("Rejection Rate:", rejected_percent)
        print("Purity:", purity_rate)
        print("Purity rate 2:", purity_rate2)
        # print("Purity rate 3:",p3)
        print("Their FDR:", 1 - true_negatives)
        print("Thier TDR:", true_positives)
        print("normal acc, recall, etc", accuracy_score(labels, preds), recall_score(labels, preds, average='macro', zero_division=0), precision_score(labels, preds, average='macro', zero_division=0))
        # print("Correctly in distribution:", true_negatives, "Correct out:", true_positives)

        # precision, recall, f1, TNR, TPR, rejection rate

        print(classification_report(excluded_labels, excluded_preds, labels=np.unique(excluded_preds), target_names=display_labels, zero_division=0))
        print("******")
        plt.title(title)
        plt.subplots_adjust(top=1.0, bottom=0.048)
        disp.plot()
        plt.title(title)
        plt.subplots_adjust(top=1.0, bottom=0.048)

        plt.show()

    if samples:
        return acc, recall, 1-true_negatives, true_positives, rejected_percent, precision, f1, purity_rate
    return acc, recall, true_negatives, true_positives

def new_evaluate(model, data, labels, show_matrix=False, excluded=False, samples=False, use_torch=False, display_labels = None):
    if use_torch:
        _, preds = torch.max(model(data), axis=1)
        mx_label = torch.max(labels)
    else:
        preds = np.argmax(model(data), axis=1)
        mx_label = np.max(labels)

    rejected_percent = np.count_nonzero(preds == mx_label) / data.shape[0]
    print("correctly out: ", np.count_nonzero(labels[preds == mx_label] == mx_label), "out of: ",
          np.count_nonzero(labels == mx_label))
    print("correctly in: ", np.count_nonzero(labels[preds != mx_label] != mx_label), "out of: ",
          np.count_nonzero(labels != mx_label))
    true_positives = np.count_nonzero(labels[preds == mx_label] == mx_label) / np.count_nonzero(labels == mx_label)
    true_negatives = np.count_nonzero(labels[preds != mx_label] != mx_label) / np.count_nonzero(labels != mx_label)

    if excluded:
        exclude = preds == mx_label
        preds = preds[~exclude]
        labels = labels[~exclude]

    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average="macro")

    if show_matrix:
        plt.rcParams.update({'font.size': 18})
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        print(cm)
        disp.plot(values_format='')
        plt.show()
    print("accuracy score: ", acc)
    print("recall score: ", recall_score(labels, preds, average="macro"))
    print("Out-of-Distribution detection: ")

    if samples:
        return acc, recall, true_negatives, true_positives, rejected_percent
    return acc, recall, true_negatives, true_positives





# old method
def evaluate_pytorch(model, data, labels, show_matrix=False, excluded_classes=False):
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    preds, true_labels = list(), list()
    for i, (inputs, targets) in enumerate(dataloader):
        y_pred = model(inputs).detach().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred.reshape((len(y_pred), 1))

        y_true = targets.numpy()
        y_true = y_true.reshape((len(y_true), 1))

        preds.append(y_pred)
        true_labels.append(y_true)
    predictions, actuals = np.vstack(preds), np.vstack(true_labels)
    if excluded_classes:
        excluded = actuals == 8
        filtered_preds = preds[~excluded]
        filtered_labels = true_labels[~excluded]
        predictions, actuals = np.vstack(filtered_preds), np.vstack(filtered_labels)

    acc = accuracy_score(actuals, predictions)

    if show_matrix:
        cm = confusion_matrix(actuals, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        print(cm)
        disp.plot()
        plt.show()
    return acc


def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


# CREDIT: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
                    title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    new_list = []
    for item in list_images:
        new_list.append(np.squeeze(item))

    list_images = new_list
    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()


def show_image(img):
    plt.imshow(np.squeeze(img), cmap='gray')


def calibrate_threshold(X, y, grads, max_runs=100, weights=(0.5, 0.5)):
    def fscore(x, y, weights):
        w1, w2 = weights
        return x * w1 + y * w2

    threshold = 10
    best_t = -1
    best_score = -1
    mx = torch.max(y)  # mx label
    prev_score = -1

    calls = 1

    reduction = False

    for i in range(max_runs):
        detected_ood = np.greater_equal(grads, threshold)

        # true positives
        tp = torch.sum(y[detected_ood] >= mx)

        # false positives
        fp = torch.sum(y[~detected_ood] < mx)

        score = fscore(tp, fp, weights)

        if score > best_score:
            best_score = score
            best_t = threshold

        if score >= prev_score:
            prev_score = score
            threshold = threshold * (1 - np.power(1 / 2, calls))
            if reduction:
                reduction = False
                calls += 1
        else:
            prev_score = score
            threshold = threshold * (1 + np.power(1 / 2, calls))

            if not reduction:
                calls += 1
                reduction = True
    return best_t


# https://stackoverflow.com/questions/48855290/plotted-confusion-matrix-values-overlapping-each-other-total-classes-90
def plot_confusion_matrix_2(cm,
                      target_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions


    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    FONT_SIZE = 16

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8*2, 6*2))    # 8, 6
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=FONT_SIZE)
        plt.yticks(tick_marks, target_names, fontsize=FONT_SIZE)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=FONT_SIZE,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=FONT_SIZE,
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()