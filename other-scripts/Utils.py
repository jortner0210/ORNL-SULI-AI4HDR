import sys
import os
import cv2
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from CAEModel import HDRClusterEncoder
from PIL import Image
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def getMnist256(image_size):
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    
    processed_mnist = np.zeros(shape=(15000, image_size[0], image_size[1], 3))
    
    for i in range(processed_mnist.shape[0]):
        image_arr = mnist_digits[i] 
        image = Image.fromarray(image_arr[:,:,0])
        if i % 2 == 0:
            rot_choices = [90, 180, 270]
            choice = random.choice(rot_choices)
            image = image.rotate(choice)

        new_img = image.resize(image_size, Image.BICUBIC)
        new_arr = np.array(new_img)
        new_arr = np.stack((new_arr, new_arr, new_arr), axis=2).astype("float32")
        new_arr /= new_arr.max()
        processed_mnist[i] = new_arr

    return processed_mnist


def maxScale(data):
    '''
    Preprocess function: scale to [0, 1]
    '''
    return data / data.max()


def loadData(data_dir: str, data_shape: tuple, preprocessFunction=None):
    # Retreive all paths in directory
    data_dir = Path(data_dir)
    image_paths = [p for p in Path(data_dir).iterdir()]
    random.shuffle(image_paths)
    data = np.zeros(shape=(len(image_paths),
                           data_shape[0],
                           data_shape[1],
                           data_shape[2]))

    data = []

    # Load each image into data
    for i in range(len(image_paths)):
        image_array = cv2.imread(str(image_paths[i]))
        image_array = cv2.resize(image_array, (data_shape[0], data_shape[1]))
        if image_array is not None:
            bad_num_check = np.isfinite(image_array)
            if bad_num_check.min() != 0:
                data.append(image_array)

    data = np.array(data)
    # Run data through user defined preprocess function if given
    if preprocessFunction is not None:
       return preprocessFunction(data) 
    else:
        return data


TEST_IMAGE_DIR      = "/mnt/data/AI4HDR/Flood/256x256/Train"
PRETRAIN_MODEL      = "pretrain-model"
PRETRAIN_TRAIN_LOSS = "pretrain_train_loss.csv"
PRETRAIN_VAL_LOSS   = "pretrain_val_loss.csv"
CLUSTER_COUNT       = 2


def compileModelResults(model_dir: str, validate_count: int, val_split=True):
    model_base_dir = Path(model_dir)
    model_dir      = model_base_dir.joinpath(PRETRAIN_MODEL)

    train_result_path    = model_base_dir.joinpath(PRETRAIN_TRAIN_LOSS)
    test_result_path = None
    if val_split:
        test_result_path     = model_base_dir.joinpath(PRETRAIN_VAL_LOSS)
    compiled_results_dir = model_base_dir.joinpath("results")

    # Create Directory for compiled results
    os.mkdir(compiled_results_dir)

    # Plot losses and save image
    train_result_array = np.loadtxt(train_result_path)
    test_result_array = None
    if val_split:
        test_result_array  = np.loadtxt(test_result_path)

    plt.plot(train_result_array, label="Training Loss")
    if val_split:
        plt.plot(test_result_array, label="Testing Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(compiled_results_dir.joinpath("loss_plot.png"))
    plt.close()

    # Predict with five images
    data = loadData(TEST_IMAGE_DIR, (256, 256, 3), maxScale)

    model = HDRClusterEncoder.loadCheckPoint(str(model_dir))
    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1.0])

    cae_output = model.cae.predict(data, verbose=0)
    encoder_output = model.encoder.predict(data, verbose=0)

    kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=20)
    y_pred = kmeans.fit_predict(encoder_output)

    if validate_count > data.shape[0]:
        validate_count = data.shape[0]

    image_prediction_dir = compiled_results_dir.joinpath("validation-images")
    if not image_prediction_dir.is_dir():
        os.mkdir(image_prediction_dir)

        for i in range(CLUSTER_COUNT):
            os.mkdir(image_prediction_dir.joinpath("{}-cluster".format(i)))

    for i in range(validate_count):
        cluster = y_pred[i]
        cv2.imwrite(str(image_prediction_dir.joinpath("{}-cluster".format(cluster)).joinpath("{}-input.png".format(i))), data[i]*255)
        cv2.imwrite(str(image_prediction_dir.joinpath("{}-cluster".format(cluster)).joinpath("{}-prediction.png".format(i))), cae_output[i]*255)


def pretrainSNE(model_dir: str, data_dir: str):
    '''
    SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    '''
    model_base_dir = Path(model_dir)
    compiled_results_dir = model_base_dir.joinpath("results")

    model_dir = model_base_dir.joinpath(PRETRAIN_MODEL)
    sne_output_dir = compiled_results_dir.joinpath("T-SNE")

    data = loadData(data_dir, (256, 256, 3), maxScale)

    model = HDRClusterEncoder.loadCheckPoint(str(model_dir))
    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1.0])

    # Reconstructions
    cae_output = model.encoder.predict(data, verbose=0)

    # Kmeans Clustering
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=20)
    y_pred = kmeans.fit_predict(cae_output)

    # T-SNE - 2D
    X_embedded = TSNE(n_components=2).fit_transform(cae_output)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred)
    plt.colorbar()
    plt.title("T-SNE - 2D")
    plt.xlabel("embedded[0]")
    plt.ylabel("embedded[1]")
    plt.savefig(compiled_results_dir.joinpath("embedded_tsne_2d.png"))
    plt.close()

    # T-SNE - 3D
    X_embedded_3d = TSNE(n_components=3).fit_transform(cae_output)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_embedded_3d[:, 0], X_embedded_3d[:, 1], X_embedded_3d[:, 2], c=y_pred)
    ax.set_title("T-SNE - 3D")
    ax.set_xlabel("embedded[0]")
    ax.set_ylabel("embedded[1]")
    ax.set_zlabel("embedded[2]")
    plt.savefig(compiled_results_dir.joinpath("embedded_tsne_3d.png"))
    plt.close()

    # PCA - 2D
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit(cae_output).transform(cae_output)

    plt.figure(figsize=(12, 10))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred)
    plt.colorbar()
    plt.title("PCA - 2D")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(compiled_results_dir.joinpath("embedded_pca_2d.png"))

    # PCA - 3D
    pca_3d = PCA(n_components=3)
    X_3d = pca_3d.fit(cae_output).transform(cae_output)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_pred)
    ax.set_title("PCA - 3D")
    ax.set_xlabel("embedded[0]")
    ax.set_ylabel("embedded[1]")
    ax.set_zlabel("embedded[2]")
    plt.savefig(compiled_results_dir.joinpath("embedded_pca_3d.png"))
    plt.close()