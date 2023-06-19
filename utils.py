import tensorflow as tf
from tensorflow.keras.losses import *
import numpy as np

def create_optimizer(optimizer_fnc, lr, other_params = {}):
    '''
    Creates one of the optimizers present in tf.keras.optimizers and returns it.

    Args --
        optimizer_func -- function | optimizer's creation function
        lr -- float or float tensor or learning_rate_function | learning rate
        other_params -- dict | default {} | contains all the arguments needed to create the optimizer
    Return --
        other_params -- tf.keras.optimizers.*
    '''
    other_params = other_params.values()
    optimizer = optimizer_fnc(lr) #, *other_params)

    return optimizer

def vgg_layers(layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model