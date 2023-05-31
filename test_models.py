import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import GOOD as g
import my_utils
import torch

lines = [
    (0, (1, 10)),
    (0, (1, 1)),
    (0, (1, 1)),
    (0, (5, 10)),
    (0, (5, 5)),
    (0, (5, 1)),

    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),

    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 1, 1, 1, 1, 1))]

display_labels_malware = np.array(["Benign", "Malware", "OOD"])
display_labels_applications = np.array(
    ["dropbox", "facebook", "google", "microsoft", "teamviewer", "twitter", "youtube", "uncategorized", "OOD"])


def load_app():

    print("Loading app known/unknown")
    data, labels = my_utils.load_known_applications()
    data_u, labels_u = my_utils.load_unknown_applications(training=False)

    # insert training from both sets
    _, data, _, labels = train_test_split(data, labels, test_size=0.20, random_state=42)

    labels_u[labels_u > -1] = np.max(labels) + 1  # should be all 8s
    print("Known size:", labels.size, "Unknown size:", labels_u.size)
    data = np.concatenate([data, data_u])
    labels = np.concatenate([labels, labels_u])

    return data, labels


def load_malware():

    print("Loading malware known/unknown")

    data, labels = my_utils.load_known_malware() #  benign - 0 , malware - 1
    data_u, labels_u = my_utils.load_unknown_malware(K1=True, training=False) #  malware(OOD) - 2
    # insert training from both sets
    _, data, _, labels = train_test_split(data, labels, test_size=0.20, random_state=42)
    _, data_u, _, labels_u = train_test_split(data_u, labels_u, test_size=0.5, random_state=42)  # 0.5 train 0.5 test

    labels_u[labels_u > -1] = np.max(labels) + 1  # should be all 2s
    print("Known size:", labels.size, "Unknown size:", labels_u.size)
    data = np.concatenate([data, data_u])
    labels = np.concatenate([labels, labels_u])

    return data, labels


def split(x, y):
    return train_test_split(x, y, test_size=0.25, random_state=42)


def plot_odin(max_predictions, argmax_predictions, labels, og_argmax_predictions, title="Malware", save_name="title"):
    mx_label = np.max(labels)
    acc, recall, tn, tp, reject, scores, purity = [], [], [], [], [], [], []
    x = []

    best_a = -1
    best_t = -1

    percent = 0
    best_purity = -1
    best_purity_t = -1
    best_purity_acc = -1
    for thresh in np.arange(0, 100, 0.01):
        ogt = thresh
        thresh = np.percentile(max_predictions, thresh)

        if thresh % 0.01 == 0:
            print("percent complete:", percent)
            percent += 1
        ood_idx = max_predictions < thresh
        new_predictions = np.copy(og_argmax_predictions)
        new_predictions[ood_idx] = mx_label

        d = my_utils.evaluate_without_model(new_predictions, labels, show_matrix=False, samples=True, use_torch=False,
                                            excluded=True)
        x.append(thresh)
        if d[0] > best_a and d[4] < 0.5:
            best_a = d[0]
            best_t = thresh

        if d[7] > best_purity and d[4] < 0.5:
            best_purity = d[7]
            best_purity_t = thresh
            best_purity_acc = d[0]

        acc.append(d[0])
        recall.append(d[1])
        tn.append(d[2])
        tp.append(d[3])
        reject.append(d[4])
        purity.append(d[7])

        ogt = 100 - ogt
        if ogt % 5 == 0:
            print("Rejected percent:", d[4])
            print("****\nacc, recall, 1-true_negatives, true_positives, rejected_percent, precision, f1, purity_rate")
            printer = [d[0], d[1], d[5], d[6], d[2], d[3], d[7]]
            for item in printer:
                print(item)
            print("****")


        # get statistics
    print("best a", best_a, "best t", best_t)

    print("best purity", best_purity, "thresh:", best_purity_t, "with acc:", best_purity_acc)

    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.title("ODIN - " + title)
    plt.rcParams["lines.linewidth"] = 5
    plt.plot(x, acc, label="accuracy", linestyle='dashed')
    # plt.plot(x, recall, label="recall", linestyle=lines[5])
    plt.plot(x, tn, label="FDR", linestyle=lines[1])
    plt.plot(x, tp, label="TDR", linestyle=lines[3])
    # plt.plot(x, tn, label="true negatives (IND detected as IND)", linestyle=lines[1])
    # plt.plot(x, tp, label="true positives(OOD detected as OOD)", linestyle=lines[3])
    plt.plot(x, reject, label="Rejection Rate", linestyle=lines[9])
    plt.plot(x, purity, label="Purity Rate", linestyle=lines[11])
    plt.rcParams.update({'font.size': 18})

    plt.legend(loc='best', fontsize="x-large")
    # plt.savefig("images/" + save_name,bbox_inches='tight')
    # plt.clf()
    plt.show()
    return best_a, best_t


def plot_bp(r, labels, grads, title="Malware", save_name="title"):
    mx = torch.max(labels)
    acc, recall, tn, tp, reject, scores, purity = [], [], [], [], [], [], []
    x = []

    best_a = -1
    best_t = -1
    best_purity = -1
    best_purity_t = -1
    best_purity_acc = -1

    for t in np.arange(0, 100, 0.1):
        ogt = t
        t = np.percentile(grads, t)
        copy_r = r.detach().clone()
        argmax_preds = torch.argmax(copy_r, axis=-1)
        argmax_preds[grads > t] = mx

        d = my_utils.evaluate_without_model(argmax_preds, labels, show_matrix=False, samples=True, excluded=True)

        if d[0] > best_a and d[4] < 0.5:
            best_a = d[0]
            best_t = t

        if d[7] > best_purity and d[4] < 0.5:
            best_purity = d[7]
            best_purity_t = t
            best_purity_acc = d[0]
        x.append(t)
        acc.append(d[0])
        recall.append(d[1])
        tn.append(d[2])
        tp.append(d[3])
        reject.append(d[4])
        purity.append(d[7])
        if ogt % 5 == 0:
            # if ogt == 70:
            #     d = my_utils.evaluate_without_model(argmax_preds, labels, show_matrix=True, samples=True,
            #                                         display_labels=display_labels_applications, excluded=True,
            #                                         title="bp_malware_cf")
            print("Rejected percent:", d[4])
            print("****\nacc, recall, 1-true_negatives, true_positives, rejected_percent, precision, f1, purity_rate")
            printer = [d[0], d[1], d[5], d[6], d[2], d[3], d[7]]
            for item in printer:
                print(item)
            print("****")

    print("best purity", best_purity, "thresh:", best_purity_t, "with acc:", best_purity_acc)

    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.title("Backpropagation - " + title)
    plt.rcParams["lines.linewidth"] = 5
    plt.plot(x, acc, label="accuracy", linestyle='dashed')
    # plt.plot(x, recall, label="recall", linestyle=lines[5])
    plt.plot(x, tn, label="FDR", linestyle=lines[1])
    plt.plot(x, tp, label="TDR", linestyle=lines[3])
    # plt.plot(x, tn, label="true negatives (IND detected as IND)", linestyle=lines[1])
    # plt.plot(x, tp, label="true positives(OOD detected as OOD)", linestyle=lines[3])
    plt.plot(x, reject, label="Rejection Rate", linestyle=lines[9])
    plt.plot(x, purity, label="Purity Rate", linestyle=lines[11])
    plt.rcParams.update({'font.size': 18})
    plt.legend(loc='best', fontsize="x-large")
    # plt.savefig("images/" + save_name,bbox_inches='tight')
    # plt.clf()
    plt.show()

    return best_a, best_t


def test_bp_malware(data, labels):
    print("TEST BP MALWARE - BEGIN")

    data = torch.tensor(data)
    labels = torch.tensor(labels)

    model = my_utils.get_malware_model("pytorch")
    grads, r = g.shadow_backpropagation(model, model.CNN[6], data)

    best_a, best_t = plot_bp(r, labels, grads, title="Malware", save_name="bp_malware")

    print("BEST ACCURACY BP MALWARE:", best_a)
    copy_r = r.detach().clone()
    argmax_preds = torch.argmax(copy_r, axis=-1)
    argmax_preds[grads > best_t] = torch.max(labels)

    d = my_utils.evaluate_without_model(argmax_preds, labels, show_matrix=True, samples=True,
                                        display_labels=display_labels_malware, excluded=True, title="bp_malware_cf")
    print("TEST BP MALWARE - END")

    return best_a


def test_bp_app(data, labels):
    print("TEST BP APP - BEGIN")

    # data, labels = load_app()
    data = torch.tensor(data)
    labels = torch.tensor(labels)

    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    model = my_utils.get_application_model("pytorch")
    grads, r = g.shadow_backpropagation(model, model.CNN[6], data)


    mx = torch.max(labels)
    best_a, best_t = plot_bp(r, labels, grads, title="Application", save_name="bp_app")

    print("BEST ACCURACY BP APP:", best_a)

    copy_r = r.detach().clone()
    argmax_preds = torch.argmax(copy_r, axis=-1)
    argmax_preds[grads > best_t] = mx

    d = my_utils.evaluate_without_model(argmax_preds, labels, show_matrix=True, samples=True,
                                        display_labels=display_labels_applications, excluded=True, title="bp_app_cf")

    print("TEST BP APP - END")
    return best_a


def test_odin_malware(data, labels, psize=0.0001):
    print("TEST ODIN MALWARE - BEGIN", psize)

    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    model = my_utils.get_malware_model("tensorflow")

    og_argmax_predictions = np.argmax(model.predict(data), axis=-1)

    # need to search ideal eps
    p_data = my_utils.create_perturbed_ds(data, model, 0.001)

    raw_predictions = model.predict(p_data)
    argmax_predictions = np.argmax(raw_predictions, axis=-1)
    max_predictions = np.max(raw_predictions, axis=-1)


    best_a, best_t = plot_odin(max_predictions, argmax_predictions, labels, og_argmax_predictions,
                               save_name="odin_malware")

    print("BEST ACCURACY ODIN MALWARE:", best_a)

    ood_idx = max_predictions < best_t
    new_predictions = np.copy(argmax_predictions)
    new_predictions[ood_idx] = np.max(labels)  # change labels to OOD when detected
    new_predictions[~ood_idx] = og_argmax_predictions[~ood_idx]

    d = my_utils.evaluate_without_model(new_predictions, labels, show_matrix=True, samples=True, use_torch=False,
                                        display_labels=display_labels_malware, excluded=True, title="odin_malware_cf")

    print("TEST ODIN MALWARE - END")
    return best_a


def test_odin_app(data, labels, psize=0.0005):
    print("TEST ODIN BEGIN - APP", psize)
    data, labels = load_app()


    model = my_utils.get_application_model("tensorflow")

    # need to search ideal eps
    og_argmax_predictions = np.argmax(model.predict(data), axis=-1)
    p_data = my_utils.create_perturbed_ds(data, model, psize)

    raw_predictions = model.predict(p_data)
    argmax_predictions = np.argmax(raw_predictions, axis=-1)
    max_predictions = np.max(raw_predictions, axis=-1)


    best_acc, best_t = plot_odin(max_predictions, argmax_predictions, labels, og_argmax_predictions,
                                 title="Application", save_name="odin_app")
    print("BEST ACCURACY ODIN APP:", best_acc)

    ood_idx = max_predictions < best_t
    new_predictions = np.copy(argmax_predictions)
    new_predictions[ood_idx] = np.max(labels)  # change labels to OOD when detected
    new_predictions[~ood_idx] = og_argmax_predictions[~ood_idx]

    my_utils.evaluate_without_model(new_predictions, labels, show_matrix=True, samples=True, use_torch=False,
                                        display_labels=display_labels_applications, excluded=True, title="odin_app_cf")
    print("TEST ODIN END")
    return best_acc



def test_abstention_malware(data, labels):
    print("K+1/ABSTENTION TEST BEGIN - MALWARE")

    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    model = my_utils.get_malware_model("tensorflow", k_plus_one=True)

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True,
                                    display_labels=display_labels_malware, use_torch=False, title="abstention_malware", excluded=True)

    print("K+1/ABSTENTION TEST END - MALWARE")


def test_abstention_app(data, labels):
    print("K+1/ABSTENTION TEST BEGIN - APPLICATION")

    model = my_utils.get_application_model("tensorflow", k_plus_one=True)

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True,
                                    display_labels=display_labels_applications, use_torch=False, title="abstention_app", excluded=True)
    print("K+1/ABSTENTION TEST END - APPLICATION")


def test_raw_with_ood():
    print("TEST BASE WITH OOD DATA- BEGIN")
    print("MALWARE")

    data, labels = load_malware()
    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    model = my_utils.get_malware_model("tensorflow")

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True,
                                    display_labels=display_labels_malware, use_torch=False, excluded=True, title="base_ood_malware_cf")

    print("APPLICATION")
    data, labels = load_app()

    model = my_utils.get_application_model("tensorflow")
    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, use_torch=False, show_matrix=True,
                                    display_labels=display_labels_applications, excluded=True, title="base_ood_app_cf")

    print("BASE WITH OOD DATA- END")


#


def test_raw():
    print("TEST RAW - BEGIN")
    print("MALWARE TF")
    data, labels = my_utils.load_known_malware()
    model = my_utils.get_malware_model("tensorflow")

    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    _, data, _, labels = split(data, labels)

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True, use_torch=False,
                                    display_labels=display_labels_malware[:-1], ood=False, title="base_malware_cf")

    print("APPLICATION TF")
    data, labels = my_utils.load_known_applications()
    _, data, _, labels = split(data, labels)

    model = my_utils.get_application_model("tensorflow")

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True, use_torch=False,
                                    display_labels=display_labels_applications[:-1], ood=False, title="base_app_cf")

    print("TEST RAW - END")





def find_odin(app_data, app_labels, mal_data, mal_labels):
    best_app = 0
    best_mal = 0
    best_p_app = 0
    best_p_mal = 0
    for psize in [0.0001,0.0005, 0.001, 0.004]:
        print("testing psize:", psize)
        temp_app = test_odin_app(app_data, app_labels, psize=psize)
        temp_mal = test_odin_malware(mal_data, mal_labels, psize=psize)

        if temp_app > best_app:
            best_app = temp_app
            best_p_app = psize
        if temp_mal > best_mal:
            best_mal = temp_mal
            best_p_mal = psize

    print("app:", best_app, best_p_app)
    print("mal:", best_mal, best_p_mal)
    exit("done")




app_data, app_labels = load_app()
mal_data, mal_labels = load_malware()
# # find_odin(app_data, app_labels, mal_data, mal_labels)
#
# # # #
# test_raw()
# test_raw_with_ood()

test_abstention_app(app_data, app_labels)
test_odin_app(app_data, app_labels)
test_bp_app(app_data, app_labels)
#
# test_abstention_malware(mal_data, mal_labels)
test_odin_malware(mal_data, mal_labels)
test_bp_malware(mal_data, mal_labels)
