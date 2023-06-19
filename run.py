import neptune.new as neptune

import tensorflow as tf
from pprint import pprint

from train_support import *
from data import *
from cost import *
# from train import *
# from data import *
from model import *
# from cost import *

from datetime import datetime
import sys
import os
import argparse

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--lr", type=float, default=0.000001,
                        help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--train-dir", type=str, default='/media/saket/Elements/datasets/feb2022demo/dataset/train',
                        help="Path to the directory containing the training dataset.")
    parser.add_argument("--dev-dir", type=str, default='/media/saket/Elements/datasets/feb2022demo/dataset/dev',
                        help="Path to the directory containing the training dataset.")
    parser.add_argument("--restore-from", type=str, default='',
                        help="Where restore model parameters from.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="starting epoch")
    parser.add_argument("--epochs", type=int, default=0,
                        help="total number of epochs")
    parser.add_argument("--save-dir", type=str, default='checlpoints/save',
                        help="dir where models will be saved")
    parser.add_argument("--dataset-name", type=str, default='Feb2022-demo',
                        help="Name of the datset")
    parser.add_argument("--loss-name", type=str, default='triplet',
                        help="Name of the loss function")
    parser.add_argument("--model-name", type=str, default='vgg-embedding-siamese',
                        help="Name of the model used")
    parser.add_argument("--optimizer-name", type=str, default='Adam',
                        help="Name of the optimizer")

    return parser.parse_args()

def train_step(X, Y, model, compute_cost, optimizer):
    '''
    Train one minibatch
    '''
    with tf.GradientTape() as tape:
        Y_pred = model(X)
        cost = compute_cost(Y, Y_pred)    
    gradient = tape.gradient(cost, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    return cost, Y_pred

def test_step(X, Y, model, compute_cost):
    '''
    Infer one minibatch
    '''
    Y_pred = model(X)
    cost = compute_cost(Y_pred)

    return cost, Y_pred

if __name__ == '__main__':
    print('###################################')
    args = get_arguments()

    ## Hyperparameters and variables
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    save_dir = args.save_dir
    start_epoch = args.start_epoch
    dataset_name = args.dataset_name
    loss_name = args.loss_name
    train_dataset_path = args.train_dir
    dev_dataset_path = args.dev_dir
    restore_model_path = args.restore_from
    model_name = args.model_name
    optimizer_name = args.optimizer_name
    print('Hyperparameters initialized')

    ## initialize neptune
    # run = neptune.init(
    #    project="10xar-saket/siamese-object-identification",
    #    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODE2ZDk5ZC00ZTZkLTQ0MmUtYTI2Zi0yOTE3MmNkODNkNmQifQ==",
    # )
    run = None

    ## Load datasets
    train_dataset = TripletLoader(batch_size, train_dataset_path ,visualize=True) 
    dev_dataset = TripletLoader(batch_size, dev_dataset_path) 

    ## Create model
    with strategy.scope():
        siamese_model = SiameseModel(EmbeddingHelper=VGG16Helper)

    # Define optimizer    
    optimizer_arg = get_optimizer_arg(optimizer_name)
    pprint(optimizer_arg)

    train_siamese(train_dataset, dev_dataset, lr, batch_size, epochs, siamese_model, 
                             train_step, test_step, triplet_cost, 
                             optimizer_name, save_dir = save_dir, 
                             checkpoint_path = restore_model_path, 
                             start_epoch = start_epoch, # eval_metrics = eval_metrics, 
                             optimizer_arg = optimizer_arg,
                             dataset_name = dataset_name, loss_name = loss_name, 
                             model_name = model_name, run = run, 
                             show_results=-1, save_image=False) 
