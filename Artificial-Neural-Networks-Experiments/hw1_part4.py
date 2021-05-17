# import necessary packages
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
import tensorflow as tf
import os
from random import randrange
from utils import part4Plots

# some parameters
LR_1 = 1e-1  # 0.1
LR_2 = 1e-2  # 0.01
LR_3 = 1e-3  # 0.001
LR = [LR_1, LR_2, LR_3]
EPOCHS = 20  # epochs
BS = 50  # batch size
IMG_W = 28  # width of the images to be trained
IMG_H = 28  # height of the images to be trained
N_CLASSES = 5  # number of classes

DATA_PATH = 'dataset/'

# load the .npy formatted data
print("[INFO] loading data...")
train_img = np.load(DATA_PATH + 'train_images.npy')
train_lbl = np.load(DATA_PATH + 'train_labels.npy')
test_img = np.load(DATA_PATH + 'test_images.npy')
test_lbl = np.load(DATA_PATH + 'test_labels.npy')

# convert data to np.array float type
train_img = np.array(train_img, dtype="float32")
train_lbl = np.array(train_lbl, dtype="int")
test_img = np.array(test_img, dtype="float32")
test_lbl = np.array(test_lbl, dtype="int")

# preprocess images [0,255] -> [-1,1]
train_img = np.true_divide(train_img, 127.5) - 1
test_img = np.true_divide(test_img, 127.5) - 1

# partition the data into training and validation splits using 90% of
# the data for training and the remaining 10% for validation
(train_img, val_img, train_lbl, val_lbl) = train_test_split(train_img, train_lbl,
                                                            test_size=0.10, stratify=train_lbl, random_state=42)
# create convolution format
train_img_conv = train_img.reshape(-1, IMG_W, IMG_H, 1)
test_img_conv = test_img.reshape(-1, IMG_W, IMG_H, 1)
val_img_conv = val_img.reshape(-1, IMG_W, IMG_H, 1)

# perform one-hot encoding on the labels
train_lbl = tf.keras.utils.to_categorical(train_lbl, N_CLASSES)
test_lbl = tf.keras.utils.to_categorical(test_lbl, N_CLASSES)
val_lbl = tf.keras.utils.to_categorical(val_lbl, N_CLASSES)

# create the PredictionLayer
PredictionLayer = tf.keras.Sequential()
PredictionLayer.add(tf.keras.layers.Dense(units=5, activation='softmax'))


def create_mlp_1(learning_rate=LR[0]):
    # construct mlp_1 model. [FC-64, ReLU] + PredictionLayer
    model_mlp_1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_W * IMG_H,)),  # input layer
        tf.keras.layers.Dense(units=64, activation='relu'),  # FC-64
        PredictionLayer  # PredictionLayer
    ])

    # create binary cross entropy loss (one-hot encoding case)
    loss_mlp_1 = tf.keras.losses.BinaryCrossentropy()

    # create optimizer
    optimizer_mlp_1 = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # compile model for training
    model_mlp_1.compile(optimizer=optimizer_mlp_1, loss=loss_mlp_1, metrics=["accuracy"])

    print("--------------MLP_1 MODEL--------------")
    print(model_mlp_1.summary())

    return model_mlp_1


def create_mlp_2(learning_rate=LR[0]):
    # construct mlp_2 model. [FC-16, ReLU, FC-64(no bias)] + PredictionLayer
    model_mlp_2 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_W * IMG_H,)),  # input layer
        tf.keras.layers.Dense(units=16, activation='relu'),  # FC-16, ReLU,
        tf.keras.layers.Dense(units=64),  # FC-64
        PredictionLayer  # PredictionLayer
    ])
    print("--------------MLP_2 MODEL--------------")
    print(model_mlp_2.summary())

    # create binary cross entropy loss (one-hot encoding case)
    loss_mlp_2 = tf.keras.losses.BinaryCrossentropy()

    # create optimizer
    optimizer_mlp_2 = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # compile model for training
    model_mlp_2.compile(optimizer=optimizer_mlp_2, loss=loss_mlp_2, metrics=["accuracy"])

    return model_mlp_2


def create_cnn_3(learning_rate=LR[0]):
    # construct cnn_3 model ‘cnn 3’ : [Conv-3×3×16, ReLU, Conv-7×7×8, ReLU,
    # MaxPool-2×2, Conv-5×5×16, MaxPool-2×2,
    model_cnn_3 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
        tf.keras.layers.Conv2D(filters=8, kernel_size=7, activation='relu'),  # Conv-7×7×8, ReLU,
        tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu'),  # Conv-7×7×8, ReLU,
        tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
    ])
    print("--------------CNN_3 MODEL--------------")
    print(model_cnn_3.summary())

    # create binary cross entropy loss (one-hot encoding case)
    loss_cnn_3 = tf.keras.losses.BinaryCrossentropy()

    # create optimizer
    optimizer_cnn_3 = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # compile model for training
    model_cnn_3.compile(optimizer=optimizer_cnn_3, loss=loss_cnn_3, metrics=["accuracy"])

    return model_cnn_3


def create_cnn_4(learning_rate=LR[0]):
    # construct cnn_4 model ‘cnn 4’ : [‘cnn 3’ : [Conv-3×3×16, ReLU,
    # Conv-5×5×8, ReLU, Conv-3×3×8, ReLU, MaxPool-2×2, Conv-5×5×16, ReLU, MaxPool-2×2,
    # GlobalAvgPool] + PredictionLayer
    model_cnn_4 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
        tf.keras.layers.Conv2D(filters=8, kernel_size=5, activation='relu'),  # Conv-5×5×8, ReLU,
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),  # Conv-3×3×8, ReLU
        tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu'),  # Conv-7×7×8, ReLU,
        tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
    ])
    print("--------------CNN_4 MODEL--------------")
    print(model_cnn_4.summary())

    # create binary cross entropy loss (one-hot encoding case)
    loss_cnn_4 = tf.keras.losses.BinaryCrossentropy()

    # create optimizer
    optimizer_cnn_4 = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # compile model for training
    model_cnn_4.compile(optimizer=optimizer_cnn_4, loss=loss_cnn_4, metrics=["accuracy"])

    return model_cnn_4


def create_cnn_5(learning_rate=LR[0], loss=None, optimizer=None):
    # construct cnn_5 model. ‘cnn 5’ : [Conv-3×3×16, ReLU, Conv-3×3×8, ReLU, Conv-3×3×8, ReLU,
    # Conv-3×3×8, ReLU, MaxPool-2×2, Conv-3×3×16, ReLU, Conv-3×3×16, ReLU, MaxPool-2×2,
    # GlobalAvgPool] + PredictionLayer
    model_cnn_5 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),  # Conv-3×3×8, ReLU,
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),  # Conv-3×3×8, ReLU
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),  # Conv-3×3×8, ReLU
        tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
        tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
    ])
    print("--------------CNN_5 MODEL--------------")
    print(model_cnn_5.summary())

    if loss == None:
        # create binary cross entropy loss (one-hot encoding case)
        loss = tf.keras.losses.BinaryCrossentropy()
    if optimizer == None:
        # create optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # compile model for training
    model_cnn_5.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model_cnn_5


# create model by name
def create_model(model_name, learning_rate):
    if model_name == "mlp_1":
        return create_mlp_1(learning_rate)
    elif model_name == "mlp_2":
        return create_mlp_2(learning_rate)
    elif model_name == "cnn_3":
        return create_cnn_3(learning_rate)
    elif model_name == "cnn_4":
        return create_cnn_4(learning_rate)
    elif model_name == "cnn_5":
        return create_cnn_5(learning_rate)
    else:
        return None


# save and load functions
def save_obj(obj, name):
    with open('part4/results/part4_' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('part4/results/part4_' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


"""****************************************************************************************************************"""
# my fit function
"""****************************************************************************************************************"""
def my_fit(model_name, x_train, y_train, x_val, y_val, epoches=20):
    # declare and initialize some parameters
    split_size = len(y_train) // BS
    split_size_val = len(y_val) // BS
    length = x_train.shape[0]
    losses = []
    val_accs = []
    val_xb = np.array_split(x_val, split_size_val)
    val_yb = np.array_split(y_val, split_size_val)
    for i in LR:
        # in each iteration, create a new model
        model = create_model(model_name, i)
        # at each iteration, we have seperate loss and accuracy curves
        loss = []
        train_acc = []
        val_acc = []
        print("------------", model_name, " LR: ", i, "------------")
        for j in range(epoches):
            print("epoch: " + str(j) + " ----------------------------> ")

            # suffle the training set
            idxs = np.arange(0, length)
            np.random.shuffle(idxs)

            x_train = x_train[idxs]
            y_train = y_train[idxs]

            # extract batches
            train_xb = np.array_split(x_train, split_size)
            train_yb = np.array_split(y_train, split_size)

            # train the batches
            index = list(range(split_size))
            for i in index:
                results = model.train_on_batch(train_xb[i], train_yb[i])
                if i % 10 == 0:  # every 10 steps
                    # record the loss
                    loss.append(results[0])
                    # record the validaditon accuracy
                    rnd_idx = randrange(split_size_val)
                    acc_val = model.evaluate(val_xb[rnd_idx], val_yb[rnd_idx])[1]
                    val_acc.append(acc_val)

        # record the loss and accuracy curves of each trial
        losses.append(np.array(loss))
        val_accs.append(np.array(val_acc))

    # convert list of np arrays to np arrays
    losses = np.array(losses)
    val_accs = np.array(val_accs)

    dic = {
        "name": model_name,
        "loss_curve_1": losses[0],
        "loss_curve_01": losses[1],
        "loss_curve_001": losses[2],
        "val_acc_curve_1": val_accs[0],
        "val_acc_curve_01": val_accs[1],
        "val_acc_curve_001": val_accs[2],
    }

    return dic


# my favorite architecture is cnn_5
dic = my_fit("cnn_5", train_img_conv, train_lbl, val_img_conv, val_lbl, 20) # 20
save_obj(dic, "cnn_5")

results = [dic]
part4Plots(results, save_dir='part4/plots', filename='part4_plot_1', show_plot=True)


"""****************************************************************************************************************"""
# Now, I will try to make scheduled learning rate to improve SGD based training
"""****************************************************************************************************************"""
def my_fit_2(model_name, x_train, y_train, x_val, y_val, LR=0.1, epoches=30):
    # declare and initialize some parameters
    split_size = len(y_train) // BS
    split_size_val = len(y_val) // BS
    length = x_train.shape[0]
    val_accs = []
    val_xb = np.array_split(x_val, split_size_val)
    val_yb = np.array_split(y_val, split_size_val)

    # create a new model cnn_5 model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss = tf.keras.losses.BinaryCrossentropy()
    model = create_cnn_5(LR, loss, optimizer)

    # at each iteration, we have seperate loss and accuracy curves
    loss = []
    val_acc = []
    print("------------", model_name, "------------")
    for j in range(epoches):
        if j == 6:
            # create optimizer, change the learning rate
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
            loss = tf.keras.losses.BinaryCrossentropy()
            # recompile model for training
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        print("epoch: " + str(j) + " ----------------------------> ")

        # suffle the training set
        idxs = np.arange(0, length)
        np.random.shuffle(idxs)

        x_train = x_train[idxs]
        y_train = y_train[idxs]

        # extract batches
        train_xb = np.array_split(x_train, split_size)
        train_yb = np.array_split(y_train, split_size)

        # train the batches
        index = list(range(split_size))
        for i in index:
            results = model.train_on_batch(train_xb[i], train_yb[i])
            if i % 10 == 0:  # every 10 steps
                # record the validation accuracy
                rnd_idx = randrange(split_size_val)
                acc_val = model.evaluate(val_xb[rnd_idx], val_yb[rnd_idx])[1]
                val_acc.append(acc_val)

    # record the accuracy curves of each trial
    val_accs.append(np.array(val_acc))

    # convert list of np arrays to np arrays
    val_accs = np.array(val_accs)

    return val_accs


val_accs = my_fit_2("cnn_5", train_img_conv, train_lbl, val_img_conv, val_lbl, 0.1, 30) # 30

dic2 = {
    "name": "cnn_5",
    "val_accs_rshp": np.mean(val_accs.reshape(-1, 10), axis=1),
    "val_accs": val_accs,
}

save_obj(dic2, "cnn_5_p2")


"""****************************************************************************************************************"""
# Repeat 2 and 3; however, in 3, continue training with 0.01 until the epoch step that you determined
# in 5. Then, set the learning rate to 0.001 and continue training until 30 epochs.
"""****************************************************************************************************************"""
def my_fit_3(model_name, x_train, y_train, x_val, y_val, LR=0.1, epoches=30):
    # declare and initialize some parameters
    split_size = len(y_train) // BS
    split_size_val = len(y_val) // BS
    length = x_train.shape[0]
    val_accs = []
    val_xb = np.array_split(x_val, split_size_val)
    val_yb = np.array_split(y_val, split_size_val)

    # create a new model cnn_5 model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss = tf.keras.losses.BinaryCrossentropy()
    model = create_cnn_5(LR, loss, optimizer)

    # at each iteration, we have seperate loss and accuracy curves
    loss = []
    val_acc = []
    print("------------", model_name, "------------")
    for j in range(epoches):
        if (j == 6):
            # create optimizer, change the learning rate
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
            loss = tf.keras.losses.BinaryCrossentropy()
            # recompile model for training
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        elif (j == 10):
            # create optimizer, change the learning rate
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
            loss = tf.keras.losses.BinaryCrossentropy()
            # recompile model for training
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        print("epoch: " + str(j) + " ----------------------------> ")

        # suffle the training set
        idxs = np.arange(0, length)
        np.random.shuffle(idxs)

        x_train = x_train[idxs]
        y_train = y_train[idxs]

        # extract batches
        train_xb = np.array_split(x_train, split_size)
        train_yb = np.array_split(y_train, split_size)

        # train the batches
        index = list(range(split_size))
        for i in index:
            results = model.train_on_batch(train_xb[i], train_yb[i])
            if i % 10 == 0:  # every 10 steps
                # record the validaditon accuracy
                rnd_idx = randrange(split_size_val)
                acc_val = model.evaluate(val_xb[rnd_idx], val_yb[rnd_idx])[1]
                val_acc.append(acc_val)

    # record theaccuracy curves of each trial
    val_accs.append(np.array(val_acc))

    # convert list of np arrays to np arrays
    val_accs = np.array(val_accs)

    return val_accs


val_accs = my_fit_3("cnn_5", train_img_conv, train_lbl, val_img_conv, val_lbl, 0.1, 30)  # 30

dic3 = {
    "name": "cnn_5_schl_3",
    "val_accs_rshp": np.mean(val_accs.reshape(-1, 10), axis=1),
    "val_accs": val_accs,
}

save_obj(dic3, "cnn_5_p3")

dic1 = load_obj("cnn_5")
dic2 = load_obj("cnn_5_p2")
dic3 = load_obj("cnn_5_p3")

curves = {
    "name": "cnn_5_scheduled_0.01",
    "loss_curve_1": (np.array([])),
    "loss_curve_01": (np.array([])),
    "loss_curve_001": (np.array([])),
    "val_acc_curve_1": (np.mean(dic1["val_acc_curve_1"].reshape(-1, 10), axis=1))[0:30],
    # np.mean(dic5["val_acc_curve_1"].reshape(-1, 10), axis=1),
    "val_acc_curve_01": (dic2["val_accs_rshp"])[0:30],
    "val_acc_curve_001": (dic3["val_accs_rshp"])[0:30],
}  # dic["val_accs_rshp"],

results = [curves]
part4Plots(results, save_dir='part4/plots', filename='part4_plot_2', show_plot=True)
