import argparse
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path

from Utils import compileModelResults, loadData, maxScale, pretrainSNE, getMnist256
from CAEModel import HDRClusterEncoder


DEFAULT_TRAIN = "/mnt/data/AI4HDR/Flood/256x256/Train"
DEFAULT_TEST  = "/mnt/data/AI4HDR/Flood/256x256/Train"
DEFAULT_MODEL_CHECKPOINT = "/mnt/data/AI4HDR/Flood/256x256/ModelCheckpoints"


if __name__ == "__main__":

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("GPU count:", len(tf.config.list_physical_devices("GPU")))

    # TO DO - OUTPUT FILE WITH COMMAND LINE ARG VALUES

    # Possible command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float, help="Learning Rate.")
    parser.add_argument('--val_split', default=0.2, type=float, help="Validation split.")
    parser.add_argument('--latent_dim', default=32, type=int, help="Size of Latent Space.")
    parser.add_argument('--clusters', default=5, type=int, help="Number of clusters.")
    parser.add_argument('--epochs', default=50, type=int, help="Number of training epochs.")
    parser.add_argument('--batch_size', default=128, type=int, help="Trainig batch size.")
    parser.add_argument('--data_dir', default=DEFAULT_TRAIN, help="Data directory.")
    parser.add_argument('--checkpoint_dir', default=DEFAULT_MODEL_CHECKPOINT, help="Directory to save model checkpoints.")
    parser.add_argument('--load_model_dir', default=None, help="Diretory to load model from.")

    args = parser.parse_args()

    print("Train on data from dir --", args.data_dir)
    print("Checkpoints saved to dir --", args.checkpoint_dir)

    # CREATE MODEL AND DISPLAY SUMMARY
    model = None
    if args.load_model_dir is not None:
        print("Loading model from dir --", args.load_model_dir)
        model = HDRClusterEncoder.loadCheckPoint(args.load_model_dir)
    else:
        print("Initializing new model")
        model = HDRClusterEncoder(lr=args.lr, latent_dim=args.latent_dim, n_clusters=args.clusters)

    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1.0])
    model.cae.summary()
    model.model.summary()

    # CREATE DATA GENERATORS
    data = loadData(args.data_dir, (256, 256, 3), preprocessFunction=maxScale)
    print("Loading Data: dataset shape=", data.shape, sep="")

    model_checkpoint_dir = Path(args.checkpoint_dir)

    # Create directory to save model and begin pretraining
    if not model_checkpoint_dir.is_dir():
        os.mkdir(model_checkpoint_dir)

        history = model.pretrain(data, 
                                 batch_size=args.batch_size, 
                                 epochs=args.epochs, 
                                 validation_split=args.val_split)

        model.saveCheckPoint(str(model_checkpoint_dir.joinpath("pretrain-model")))
        
        # Save pretrain results
        np.savetxt("{}/pretrain_train_loss.csv".format(str(model_checkpoint_dir)), 
                   np.array(history.history["loss"]), 
                   delimiter=",")
        if not args.val_split == 0.0:
            np.savetxt("{}/pretrain_val_loss.csv".format(str(model_checkpoint_dir)), 
                       np.array(history.history["val_loss"]), 
                       delimiter=",")

    else:
        print("Model Dir: {} -- Already Exists".format(str(model_checkpoint_dir)))

    out_file = open("{}/model-description.csv".format(str(model_checkpoint_dir)), "w")

    out_file.write("learning_rate,val_split,latent_dim,clusters,epochs,batch_size,\n")
    out_file.write("{},{},{},{},{},{},\n".format(
        args.lr,
        args.val_split,
        args.latent_dim,
        args.clusters,
        args.epochs,
        args.batch_size
    ))

    out_file.close()

    if args.val_split == 0.0:
        compileModelResults(str(model_checkpoint_dir), 1000, val_split=False)
    else:
        compileModelResults(str(model_checkpoint_dir), 1000, val_split=True)
    pretrainSNE(str(model_checkpoint_dir), args.data_dir)