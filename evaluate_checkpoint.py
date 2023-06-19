from builtins import print
from os import makedirs
from xml.dom.minidom import Identified
import tensorflow as tf
from model import *

import sys
import os
import argparse

from matplotlib import pyplot as plt
import cv2
import numpy as np

import pickle
import time

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--input_image_dir", type=str, default='../../Datasets/barcode_data/medium_barcode_1d/BarcodeDatasets/Dataset1',
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--checkpoint_path", type=str, default='',
                        help="Where restore model parameters from.")
    # parser.add_argument("--emb_db_path", type=str, default='',
    #                    help="Where restore model parameters from.")
    

    return parser.parse_args()

def preprocess_input(input_image, shape=(224, 224), normalize=True, dtype=np.float32):
    '''
    Reads and preprocesses the input image.
    Arguments ::
        input_image -- ndarray or str | image to be normalized or path to the image
        shape -- tupple | shape to resize the input image in the form (w, h) | default None
                    | if None, does not resize the input image
        normalize -- bool | if set True, normalizes the input image
                        | default False
        dtype -- datatype of the input to the model | default np.float32
    Returns ::
        input_image -- ndarray | preprocessed input image
    '''
    if isinstance(input_image, str):
        ## Read the image
        input_image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB)

    if dtype == np.int8:
        return input_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(dtype)

    if np.max(input_image) > 2:
        input_image = input_image / 255.

    if shape != None:
        input_image = cv2.resize(input_image, shape)

    input_image = input_image[np.newaxis, :, :, :].astype(np.float32)

    return input_image

@tf.function
def test_step(X, model):
    '''
    Train one minibatch
    '''
    embeddings = model(X)

    return embeddings


def get_embeddings(input_image_path, model):
    '''
    This function infers one image and saves the output.
    Arguments --
        input_image_path -- str | input image path
        model - tf.keras.lauers.Model | pretrained model
    '''
    ## Pre-process image
    input_image = preprocess_input(input_image_path, shape=(224, 224))

    ## Infer
    embeddings = test_step(input_image, model).numpy()

    return embeddings

def identify_image(input_image_path, class_embeddings, model):
    '''
    Given an input image and class embeddings this function compates the embeddings of the
    input image and compares it with all the embeddings present in the embedding database
    and assigns a class to it.

    Arguments --
        input_image_path -- str | path where the input image is stores
        class_embeddings -- dict | contains class names as keys and embeddings as values
        model - tf.keras.lauers.Model | pretrained model
    Retuen --
        identified_class -- str | name of the class that will be assigned to the image
    '''
    # start_time = time.time()
    embeddings = get_embeddings(input_image_path, model)

    identified_class = ''
    dist = 0

    dist_dict = {}
    i = 0
    for class_name in class_embeddings:
        class_dist = (np.mean(np.sum((class_embeddings[class_name] - embeddings) ** 2)))
        dist_dict[class_name] = class_dist

    identified_class = min(dist_dict, key=dist_dict.get)
    dist = dist_dict[identified_class]
    identified_class = identified_class if dist < 1. else ''
    # print(input_image_path, dist, identified_class)
    # print(dist_dict)
    # print(f'time taken = {time.time() - start_time}')
    return identified_class

def evaluate_image_dir(input_image_dir, class_embeddings, model, debug=False):
    '''
    Given an input image dir and class embeddings this function compates the embeddings of all the
    images in the dir and compares it with all the embeddings present in the embedding database
    and assigns a class to it.

    Arguments --
        input_image_dir -- str | path where the input images are stored
        class_embeddings -- dict | contains class names as keys and embeddings as values
        model - tf.keras.lauers.Model | pretrained model
    '''
    debug_dir = './debug'
    n_classes = 0
    accuracy = 0
    if debug:
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
    for image_name in os.listdir(input_image_dir):
        if os.path.isdir(os.path.join(input_image_dir, image_name)):
            correct = 0
            class_name = image_name
            class_dir = os.path.join(debug_dir, class_name)
            pos_class_dir = os.path.join(class_dir, 'pos')
            neg_class_dir = os.path.join(class_dir, 'neg')
            if debug:
                if not os.path.exists(pos_class_dir):
                    os.makedirs(pos_class_dir)
                if not os.path.exists(neg_class_dir):
                    os.makedirs(neg_class_dir)
            for item_name in os.listdir(os.path.join(input_image_dir, class_name)):
                identified_class = identify_image(os.path.join(os.path.join(input_image_dir, class_name), item_name), class_embeddings, model)
                # print(f'{image_name} = {identified_class}')
                if image_name == identified_class:
                    correct += 1
                    if debug:
                        os.system(f'cp {os.path.join(os.path.join(input_image_dir, class_name), item_name)} {pos_class_dir}')
                else:
                    if debug:
                        os.system(f'cp {os.path.join(os.path.join(input_image_dir, class_name), item_name)} {neg_class_dir}')
            print(f'####################### Accuracy for {class_name} =  {correct / len(os.listdir(os.path.join(input_image_dir, class_name)))} ################################')
            accuracy += correct / len(os.listdir(os.path.join(input_image_dir, class_name)))
            n_classes += 1
        else:
            identified_class = identify_image(os.path.join(input_image_dir, image_name), class_embeddings, model)
            print(f'{image_name} = {identified_class}')
    if n_classes > 0:
        accuracy /= n_classes
        print(f'%%%%%%%%%%%%%%%% Accuracy Total = {accuracy} %%%%%%%%%%%%%%%%%%%%%')
    
if __name__ == '__main__':
    ## get all the arguments
    args = get_arguments()
    input_image_dir = args.input_image_dir
    pickle_path = '' # args.output_image_path
    checkpoint_path = args.checkpoint_path

    ## reset tf graph
    tf.keras.backend.clear_session()    

    ## allow gpu growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    gpus = ['gpu:'+gpu.name[-1] for gpu in gpus]
    print(f'GPUs : {gpus}')

    ## instantiate the model
    print('Loading Model...', end='\r')
    model = SiameseModel(training=False)

    ## load checkpoint
    checkpoint = tf.train.Checkpoint(net=model)
    if checkpoint_path != '':
        if os.path.exists(checkpoint_path):
            print(f'Restoring checkpoint from {checkpoint_path}', end='\r')
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
            sys.stdout.write("\033[K")
            print(f'Restored checkpoint from {checkpoint_path}')
        else:
            print(f'Checkpoint {checkpoint_path} does not exist')

    with open('embedding_db.pickle', 'rb') as handle:
         class_embeddings = pickle.load(handle)
         
    ## 4. call the infer function
    if os.path.isdir(input_image_dir):
        evaluate_image_dir(input_image_dir, class_embeddings, model, debug=True)
        




