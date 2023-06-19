from builtins import print
from copyreg import pickle
import tensorflow as tf
from model import *

import sys
import os
import argparse

from matplotlib import pyplot as plt
import cv2
import numpy as np

import pickle

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--input_class_path", type=str, default='../../Datasets/barcode_data/medium_barcode_1d/BarcodeDatasets/Dataset1',
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--checkpoint_path", type=str, default='',
                        help="Where restore model parameters from.")

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
        print(input_image)
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


def get_embeddings_sigle(input_image_path, model):
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
    print(embeddings.shape)

    return embeddings

def get_embeddings_dir(input_image_dir, model):
    '''
    This function infers all the images present in a dir and saves the output.
    Arguments --
        input_image_path -- str | input image path
        model - tf.keras.lauers.Model | pretrained model
    '''
    print(input_image_dir)
    embeddings = 0
    for image_name in os.listdir(input_image_dir):
        embeddings += get_embeddings_sigle(os.path.join(input_image_dir, image_name), model)
    
    return embeddings / len(os.listdir(input_image_dir))
  

def get_embeddings_dict(class_dir, model):
    '''
    Given a class dir of dir (where the inner dirs belong to each class)
    this function computes the embeddings for each of them, stores them in a
    dict and returns them.


    Args --
        class_dir -- str | dir where the classes are stored in a dir structure
                            | i.e. PARRENT_DIR
                                        \___CLASS_1
                                                \___IM1
                                                |___IM2
                                                    ...
                                                |___IM5
                                        \___CLASS2
                                            and so on
        model -- tf.keras.lauers.Model | pretrained model
    '''
    embeddings_dict = {k : get_embeddings_dir(os.path.join(class_dir, k), model) for k in os.listdir(class_dir)}

    return embeddings_dict

if __name__ == '__main__':
    ## get all the arguments
    args = get_arguments()
    input_class_path = args.input_class_path
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

    ## 4. call the infer function
    if os.path.isdir(input_class_path):
        embeddings_dict = get_embeddings_dict(input_class_path, model)
        with open('embedding_db.pickle', 'wb') as handle:
            pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)