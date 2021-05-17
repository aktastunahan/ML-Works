# import necessery packages
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
import pickle

from utils import part3Plots

# INIT_LR = 1e-4 # initial learning rate
EPOCHS = 15  # epochs
BS = 50  # batch size
IMG_W = 28  # width of the images to be trained
IMG_H = 28  # height of the images to be trained
N_CLASSES = 5  # number of classes
IMG_W = 28
IMG_H = 28

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
train_img = (train_img / 127.5) - 1
test_img = (test_img / 127.5) - 1

# partition the data into training and validation splits using 90% of
# the data for training and the remaining 10% for validation
(train_img, val_img, train_lbl, val_lbl) = train_test_split(train_img, train_lbl,
                                                            test_size=0.10, stratify=train_lbl, random_state=42)

# reformat images for cnn
cnn_train_img = train_img.reshape(-1, IMG_W, IMG_H, 1)

# perform one-hot encoding on the labels
train_lbl = tf.keras.utils.to_categorical(train_lbl, N_CLASSES)
test_lbl = tf.keras.utils.to_categorical(test_lbl, N_CLASSES)
val_lbl = tf.keras.utils.to_categorical(val_lbl, N_CLASSES)

# create the PredictionLayer
PredictionLayer = tf.keras.Sequential()
PredictionLayer.add(tf.keras.layers.Dense(units=5, activation='softmax'))

# construct mlp_1 model. [FC-64, ReLU] + PredictionLayer
model_mlp_1_relu = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W * IMG_H,)),  # input layer
    tf.keras.layers.Dense(units=64, activation='relu'),  # FC-64
    PredictionLayer  # PredictionLayer
])
print("--------------MLP_1 MODEL(RELU)--------------")
print(model_mlp_1_relu.summary())

model_mlp_1_sigm = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W * IMG_H,)),  # input layer
    tf.keras.layers.Dense(units=64, activation='sigmoid'),  # FC-64
    PredictionLayer  # PredictionLayer
])
print("--------------MLP_1 MODEL(SIGMOID)--------------")
print(model_mlp_1_sigm.summary())

# construct mlp_2 model. [FC-16, ReLU, FC-64(no bias)] + PredictionLayer
model_mlp_2_relu = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W * IMG_H,)),  # input layer
    tf.keras.layers.Dense(units=16, activation='relu'),  # FC-16, ReLU,
    tf.keras.layers.Dense(units=64),  # FC-64
    PredictionLayer  # PredictionLayer
])
print("--------------MLP_2 MODEL(RELU)--------------")
print(model_mlp_2_relu.summary())

model_mlp_2_sigm = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W * IMG_H,)),  # input layer
    tf.keras.layers.Dense(units=16, activation='sigmoid'),  # FC-16, ReLU,
    tf.keras.layers.Dense(units=64),  # FC-64
    PredictionLayer  # PredictionLayer
])
print("--------------MLP_2 MODEL(SIGMOID)--------------")
print(model_mlp_2_sigm.summary())

# construct cnn_3 model ‘cnn 3’ : [Conv-3×3×16, ReLU, Conv-7×7×8, ReLU,
# MaxPool-2×2, Conv-5×5×16, MaxPool-2×2,
model_cnn_3_relu = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
    tf.keras.layers.Conv2D(filters=8, kernel_size=7, activation='relu'),  # Conv-7×7×8, ReLU,
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu'),  # Conv-5×5×16, ReLU,
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
])
print("--------------CNN_3 MODEL(RELU)--------------")
print(model_cnn_3_relu.summary())

model_cnn_3_sigm = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='sigmoid'),  # Conv-3×3×16, Relu,
    tf.keras.layers.Conv2D(filters=8, kernel_size=7, activation='sigmoid'),  # Conv-7×7×8, ReLU,
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),  # Conv-7×7×8, ReLU,
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
])
print("--------------CNN_3 MODEL(SIGM)--------------")
print(model_cnn_3_sigm.summary())

# construct cnn_4 model ‘cnn 4’ : [‘cnn 3’ : [Conv-3×3×16, ReLU,
# Conv-5×5×8, ReLU, Conv-3×3×8, ReLU, MaxPool-2×2, Conv-5×5×16, ReLU, MaxPool-2×2,
# GlobalAvgPool] + PredictionLayer
model_cnn_4_relu = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
    tf.keras.layers.Conv2D(filters=8, kernel_size=5, activation='relu'),  # Conv-5×5×8, ReLU,
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),  # Conv-3×3×8, ReLU
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu'),  # Conv-5×5×16, ReLU,
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
])
print("--------------CNN_4 MODEL(RELU)--------------")
print(model_cnn_4_relu.summary())

model_cnn_4_sigm = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='sigmoid'),  # Conv-3×3×16, Relu,
    tf.keras.layers.Conv2D(filters=8, kernel_size=5, activation='sigmoid'),  # Conv-5×5×8, ReLU,
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='sigmoid'),  # Conv-3×3×8, ReLU
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),  # Conv-7×7×8, ReLU,
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
])
print("--------------CNN_4 MODEL(SIGMOID)--------------")
print(model_cnn_4_sigm.summary())

# construct cnn_5 model. ‘cnn 5’ : [Conv-3×3×16, ReLU, Conv-3×3×8, ReLU, Conv-3×3×8, ReLU,
# Conv-3×3×8, ReLU, MaxPool-2×2, Conv-3×3×16, ReLU, Conv-3×3×16, ReLU, MaxPool-2×2,
# GlobalAvgPool] + PredictionLayer
model_cnn_5_relu = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),  # Conv-5×5×8, ReLU,
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),  # Conv-3×3×8, ReLU
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),  # Conv-3×3×8, ReLU
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),  # Conv-3×3×16, Relu,
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
])
print("--------------CNN_5 MODEL (RELU)--------------")
print(model_cnn_5_relu.summary())

model_cnn_5_sigm = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_W, IMG_H, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='sigmoid'),  # Conv-3×3×16, Relu,
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='sigmoid'),  # Conv-5×5×8, ReLU,
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='sigmoid'),  # Conv-3×3×8, ReLU
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='sigmoid'),  # Conv-3×3×8, ReLU
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='sigmoid'),  # Conv-3×3×16, Relu,
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='sigmoid'),  # Conv-3×3×16, Relu,
    tf.keras.layers.MaxPool2D((2, 2)),  # MaxPool-2×2
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax', use_bias='false')  # prediction layer
])
print("--------------CNN_5 MODEL(SIGM)--------------")
print(model_cnn_5_sigm.summary())

# create binary cross entropy loss (one-hot encoding case)
loss_mlp_1_relu = tf.keras.losses.BinaryCrossentropy()
loss_mlp_2_relu = tf.keras.losses.BinaryCrossentropy()
loss_cnn_3_relu = tf.keras.losses.BinaryCrossentropy()
loss_cnn_4_relu = tf.keras.losses.BinaryCrossentropy()
loss_cnn_5_relu = tf.keras.losses.BinaryCrossentropy()

# create optimizer
optimizer_mlp_1_relu = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_mlp_2_relu = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_cnn_3_relu = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_cnn_4_relu = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_cnn_5_relu = tf.keras.optimizers.SGD(learning_rate=0.01)

# compile model for training
model_mlp_1_relu.compile(optimizer=optimizer_mlp_1_relu, loss=loss_mlp_1_relu, metrics=["accuracy"])
model_mlp_2_relu.compile(optimizer=optimizer_mlp_2_relu, loss=loss_mlp_2_relu, metrics=["accuracy"])
model_cnn_3_relu.compile(optimizer=optimizer_cnn_3_relu, loss=loss_cnn_3_relu, metrics=["accuracy"])
model_cnn_4_relu.compile(optimizer=optimizer_cnn_4_relu, loss=loss_cnn_4_relu, metrics=["accuracy"])
model_cnn_5_relu.compile(optimizer=optimizer_cnn_5_relu, loss=loss_cnn_5_relu, metrics=["accuracy"])

# create binary cross entropy loss (one-hot encoding case)
loss_mlp_1_sigm = tf.keras.losses.BinaryCrossentropy()
loss_mlp_2_sigm = tf.keras.losses.BinaryCrossentropy()
loss_cnn_3_sigm = tf.keras.losses.BinaryCrossentropy()
loss_cnn_4_sigm = tf.keras.losses.BinaryCrossentropy()
loss_cnn_5_sigm = tf.keras.losses.BinaryCrossentropy()

# create optimizer
optimizer_mlp_1_sigm = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_mlp_2_sigm = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_cnn_3_sigm = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_cnn_4_sigm = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_cnn_5_sigm = tf.keras.optimizers.SGD(learning_rate=0.01)

# compile model for training
model_mlp_1_sigm.compile(optimizer=optimizer_mlp_1_sigm, loss=loss_mlp_1_sigm, metrics=["accuracy"])
model_mlp_2_sigm.compile(optimizer=optimizer_mlp_2_sigm, loss=loss_mlp_2_sigm, metrics=["accuracy"])
model_cnn_3_sigm.compile(optimizer=optimizer_cnn_3_sigm, loss=loss_cnn_3_sigm, metrics=["accuracy"])
model_cnn_4_sigm.compile(optimizer=optimizer_cnn_4_sigm, loss=loss_cnn_4_sigm, metrics=["accuracy"])
model_cnn_5_sigm.compile(optimizer=optimizer_cnn_5_sigm, loss=loss_cnn_5_sigm, metrics=["accuracy"])


# training function
def my_fit(model, x_train, y_train, iteration=1, epoches=15):
    split_size = len(y_train) // BS
    length = x_train.shape[0]
    weights = model.trainable_weights[0].numpy()
    losses = []
    grads = []
    for i in range(iteration):
        print('iteration: ', (i + 1))
        for j in range(epoches):
            print("epoch: " + str(j) + " ----------------------------> ")

            idxs = np.arange(0, length)
            np.random.shuffle(idxs)

            x_train = x_train[idxs]
            y_train = y_train[idxs]
            xb = np.array_split(x_train, split_size)
            yb = np.array_split(y_train, split_size)

            acc = 0
            index = list(range(split_size))
            for i in index:
                results = model.train_on_batch(xb[i], yb[i])
                acc += results[1]
                if i % 10 == 0:
                    losses.append(results[0])
                    weights_new = model.trainable_weights[0].numpy()
                    grad = (weights_new - weights) / 0.01
                    grads.append(np.linalg.norm(grad))
                    weights = weights_new

            acc = acc / split_size
            print("accuracy: ", acc)

    dictionary = {
        "losses": losses,
        "grads": grads,
    }

    return dictionary


def save_obj(obj, name):
    with open('part3/results/part3_' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('part3/results/part3_' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


dic_mlp_1_relu = my_fit(model_mlp_1_relu, train_img, train_lbl)
dic_mlp_1_sigm = my_fit(model_mlp_1_sigm, train_img, train_lbl)
dic_mlp_2_relu = my_fit(model_mlp_2_relu, train_img, train_lbl)
dic_mlp_2_sigm = my_fit(model_mlp_2_sigm, train_img, train_lbl)
dic_cnn_3_relu = my_fit(model_cnn_3_relu, cnn_train_img, train_lbl)
dic_cnn_3_sigm = my_fit(model_cnn_3_sigm, cnn_train_img, train_lbl)
dic_cnn_4_relu = my_fit(model_cnn_4_relu, cnn_train_img, train_lbl)
dic_cnn_4_sigm = my_fit(model_cnn_4_sigm, cnn_train_img, train_lbl)
dic_cnn_5_relu = my_fit(model_cnn_5_relu, cnn_train_img, train_lbl)
dic_cnn_5_sigm = my_fit(model_cnn_5_sigm, cnn_train_img, train_lbl)


def to_result_dic(name, dic_relu, dic_sigm):
    result = {
        "name": name,
        "relu_loss_curve": dic_relu["losses"],
        "sigmoid_loss_curve": dic_sigm["losses"],
        "relu_grad_curve": dic_relu["grads"],
        "sigmoid_grad_curve": dic_sigm["grads"]
    }
    return result


# convert the results to a suitable format to print
dic_mlp_1 = to_result_dic("mlp_1", dic_mlp_1_relu, dic_mlp_1_sigm)
dic_mlp_2 = to_result_dic("mlp_2", dic_mlp_2_relu, dic_mlp_2_sigm)
dic_cnn_3 = to_result_dic("cnn_3", dic_cnn_3_relu, dic_cnn_3_sigm)
dic_cnn_4 = to_result_dic("cnn_4", dic_cnn_4_relu, dic_cnn_4_sigm)
dic_cnn_5 = to_result_dic("cnn_5", dic_cnn_5_relu, dic_cnn_5_sigm)
save_obj(dic_mlp_1, "mlp_1")
save_obj(dic_mlp_2, "mlp_2")
save_obj(dic_cnn_3, "cnn_3")
save_obj(dic_cnn_4, "cnn_4")
save_obj(dic_cnn_5, "cnn_5")


# Reduces the shape (x,) to (x/10,) by taking average of each 10 sample.
def reduce_graph_noise(dictionary, avr=10):
    result = {
        "name": dictionary["name"],
        "relu_loss_curve": np.mean(np.array(dictionary["relu_loss_curve"]).reshape(-1, avr), axis=1),
        "sigmoid_loss_curve": np.mean(np.array(dictionary["sigmoid_loss_curve"]).reshape(-1, avr), axis=1),
        "relu_grad_curve": np.mean(np.array(dictionary["relu_loss_curve"]).reshape(-1, avr), axis=1),
        "sigmoid_grad_curve": np.mean(np.array(dictionary["sigmoid_loss_curve"]).reshape(-1, avr), axis=1)
    }
    return result


# reduce the noise of curves
dic_mlp_1 = reduce_graph_noise(dic_mlp_1)
dic_mlp_2 = reduce_graph_noise(dic_mlp_2)
dic_cnn_3 = reduce_graph_noise(dic_cnn_3)
dic_cnn_4 = reduce_graph_noise(dic_cnn_4)
dic_cnn_5 = reduce_graph_noise(dic_cnn_5)

results = [ dic_mlp_1, dic_mlp_2, dic_cnn_3, dic_cnn_4, dic_cnn_5]
part3Plots(results, save_dir='part3/plots/', filename='part3_graph', show_plot=True)
