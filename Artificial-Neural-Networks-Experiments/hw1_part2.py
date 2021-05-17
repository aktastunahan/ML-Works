# import necessery packages
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
import os
import pickle
from random import randrange
from utils import part2Plots, visualizeWeights

# some parameters
INIT_LR = 1e-2  # initial learning rate
EPOCHS = 15  # epochs
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
# create convolution formatted images
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


# ****************** MODEL CREATION FUNCTIONS ****************** #
def create_mlp_1():
    # construct mlp_1 model. [FC-64, ReLU] + PredictionLayer
    model_mlp_1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_W * IMG_H,)),  # input layer
        tf.keras.layers.Dense(units=64, activation='relu'),  # FC-64
        PredictionLayer  # PredictionLayer
    ])

    # create binary cross entropy loss (one-hot encoding case)
    loss_mlp_1 = tf.keras.losses.BinaryCrossentropy()

    # create optimizer
    optimizer_mlp_1 = tf.keras.optimizers.SGD(learning_rate=INIT_LR)

    # compile model for training
    model_mlp_1.compile(optimizer=optimizer_mlp_1, loss=loss_mlp_1, metrics=["accuracy"])

    print("--------------MLP_1 MODEL--------------")
    print(model_mlp_1.summary())

    return model_mlp_1


def create_mlp_2():
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
    optimizer_mlp_2 = tf.keras.optimizers.SGD(learning_rate=INIT_LR)

    # compile model for training
    model_mlp_2.compile(optimizer=optimizer_mlp_2, loss=loss_mlp_2, metrics=["accuracy"])

    return model_mlp_2


def create_cnn_3():
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
    optimizer_cnn_3 = tf.keras.optimizers.SGD(learning_rate=INIT_LR)

    # compile model for training
    model_cnn_3.compile(optimizer=optimizer_cnn_3, loss=loss_cnn_3, metrics=["accuracy"])

    return model_cnn_3


def create_cnn_4():
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
    optimizer_cnn_4 = tf.keras.optimizers.SGD(learning_rate=INIT_LR)

    # compile model for training
    model_cnn_4.compile(optimizer=optimizer_cnn_4, loss=loss_cnn_4, metrics=["accuracy"])

    return model_cnn_4


def create_cnn_5():
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

    # create binary cross entropy loss (one-hot encoding case)
    loss_cnn_5 = tf.keras.losses.BinaryCrossentropy()

    # create optimizer
    optimizer_cnn_5 = tf.keras.optimizers.SGD(learning_rate=INIT_LR)

    # compile model for training
    model_cnn_5.compile(optimizer=optimizer_cnn_5, loss=loss_cnn_5, metrics=["accuracy"])

    return model_cnn_5

    # create model by name


def create_model(model_name):
    if model_name == "mlp_1":
        return create_mlp_1()
    elif model_name == "mlp_2":
        return create_mlp_2()
    elif model_name == "cnn_3":
        return create_cnn_3()
    elif model_name == "cnn_4":
        return create_cnn_4()
    elif model_name == "cnn_5":
        return create_cnn_5()
    else:
        return None


# load and save functions
def save_obj(obj, name):
    with open('part2/results/part2_' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('part2/results/part2_' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# my fit function that uses train_on_batch()
def my_fit(model_name, x_train, y_train, x_test, y_test,
           x_val, y_val, iteration=1, epoches=15):
    # declare and initialize some parameters
    split_size = len(y_train) // BS
    split_size_val = len(y_val) // BS
    length = x_train.shape[0]
    losses = []
    train_accs = []
    test_accs = []
    val_accs = []
    weights = []
    # split validation dataset into bacthes
    val_xb = np.array_split(x_val, split_size_val)
    val_yb = np.array_split(y_val, split_size_val)
    for i in range(iteration):
        # in each iteration, create a new model
        model = create_model(model_name)
        # at each iteration, we have seperate loss and accuracy curves
        loss = []
        train_acc = []
        val_acc = []
        print("------------", model_name, " iteration: ", (i + 1), "------------")
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
                    # record the accuracy
                    train_acc.append(results[1])
                    # record the validaditon accuracy
                    rnd_idx = randrange(split_size_val)
                    acc_val = model.evaluate(val_xb[rnd_idx], val_yb[rnd_idx])[1]
                    val_acc.append(acc_val)

        # record the loss and accuracy curves of each trial
        losses.append(np.array(loss))
        train_accs.append(np.array(train_acc))
        val_accs.append(np.array(val_acc))
        weights.append(model.trainable_weights[0].numpy())
        # Compute test accuracy
        test_accs.append(model.evaluate(x_test, y_test)[1])

    # convert list of np arrays to np arrays
    losses = np.array(losses)
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    val_accs = np.array(val_accs)
    weights = np.array(weights)

    best_idx = np.argmax(test_accs)
    dic = {
        "name": model_name,
        "loss_curve": np.mean(losses, axis=0),
        "train_acc_curve": np.mean(train_accs, axis=0),
        "val_acc_curve": np.mean(val_accs, axis=0),
        "test_acc": test_accs[best_idx],
        "weights": weights[best_idx]
    }

    return dic


# train mlp_1 model and save the results
dic1 = my_fit("mlp_1", train_img, train_lbl, test_img, test_lbl, val_img, val_lbl)
save_obj(dic1, "mlp_1")

# train mlp_2 model and save the results
dic2 = my_fit("mlp_2", train_img, train_lbl, test_img, test_lbl, val_img, val_lbl)
save_obj(dic2, "mlp_2")

# train cnn_3 model and save the results
dic3 = my_fit("cnn_3", train_img_conv, train_lbl, test_img_conv, test_lbl,
              val_img_conv, val_lbl)
save_obj(dic3, "cnn_3")

# train cnn_4 model and save the results
dic4 = my_fit("cnn_4", train_img_conv, train_lbl, test_img_conv, test_lbl,
              val_img_conv, val_lbl)
save_obj(dic4, "cnn_4")

# train cnn_5 model and save the results
dic5 = my_fit("cnn_5", train_img_conv, train_lbl, test_img_conv, test_lbl,
              val_img_conv, val_lbl)
save_obj(dic5, "cnn_5")

# draw the curves
results = [dic1, dic2, dic3, dic4, dic5]
part2Plots(results, save_dir='part2/plots/', filename='part2_plot', show_plot=True)

# visualize the weights
visualizeWeights(dic1['weights'], save_dir='part2/plots', filename="mlp_1_weights")
visualizeWeights(dic2['weights'], save_dir='part2/plots', filename="mlp_2_weights")
visualizeWeights(dic3['weights'], save_dir='part2/plots', filename="cnn_3_weights")
visualizeWeights(dic4['weights'], save_dir='part2/plots', filename="cnn_4_weights")
visualizeWeights(dic5['weights'], save_dir='part2/plots', filename="cnn_5_weights")