import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import os
import sys
import math
import random
import pickle as p
from collections import defaultdict

# In[2]:

def preprocess_input(input_image, shape=(512, 512), normalize=True, dtype=np.float32):
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
        image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB)

    if dtype == np.int8:
        return input_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(dtype)

    gr_image =  cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    input_image = np.ones_like(input_image) * gr_image

    if np.max(input_image) > 2:
        input_image = input_image / 255.

    if shape != None:
        input_image = cv2.resize(input_image, shape)

    input_image = input_image[np.newaxis, :, :, :].astype(np.float32)

    return input_image


# create minibatches
def random_mini_batches(X_train, Y_train, minibatch_size):
    '''
    Divides the entire trainig set into minibatches of size minibatch size randomly.
    
    args ->
    X_train : training set features numpy array
    Y_train : training set labels numppy arrray
    
    return ->
    minibatches : a tupple containing all the randomly choosen minibatches from X_train and Y_train
    '''
    minibatches = []
    m = X_train.shape[0]         # number of training examples
    indices = random.sample(range(m), m)     # indices of the minibatches
    
    # divide the training set into minibatches
    while len(indices) > 0:
        minibatch_X = X_train[:minibatch_size]
        minibatch_Y = Y_train[:minibatch_size]
        del indices[:minibatch_size]    # remove the used indices/
        minibatches.append((minibatch_X, minibatch_Y))
        
    return minibatches


def random_mini_batches_from_SSD(X_train, Y_train, minibatch_size):
    '''
    Divides the entire trainig set into minibatches of size minibatch size randomly.
    
    args ->
    X_train : python list containing all the image dirs
    Y_train : python list containing all the image dirs
    
    return ->
    minibatches : a tupple containing all the randomly choosen minibatches from X_train and Y_train
    '''
    minibatches = []
    m = len(X_train)         # number of training examples
    start = 0
    end = minibatch_size
    minibatch_X_names = []
    minibatch_Y_names = []
    minibatches = []
    indices = list(range(m))
    
    while m > 0:
        minibatch_X_names = X_train[start : end]
        minibatch_Y_names = Y_train[start : end]
        if m < minibatch_size:
            minibatch_size = m
        m -= minibatch_size
        start = end
        end += minibatch_size
        minibatches.append((minibatch_X_names, minibatch_Y_names))
        
    return minibatches

def random_mini_batches_from_SSD_tripple(anchor_train, possitive_train, negetive_train, labels, minibatch_size):
    '''
    Divides the entire trainig set into minibatches of size minibatch size randomly.
    This function should be used for training siamese networks, where you need "anchor",
    "possitive" and "negetive" samples to complete 1 record.
    
    args ->
    anchor_train : python list containing all the image dirs
    possitive_train : python list containing all the image dirs
    negetive_train : python list containing all the image dirs
    labels : labels
    
    return ->
    minibatches : a tupple containing all the randomly choosen minibatches from X_train(tripplet tupple)
                                     and Y_train
    '''
    minibatches = []
    m = len(anchor_train)         # number of training examples
    start = 0
    end = minibatch_size
    minibatches = []
    ziped = list(zip(anchor_train, possitive_train, negetive_train, labels))
    random.shuffle(ziped)
    
    while m > 0:
        zipped_batch = ziped[start : end]
        unzipped_batch = [
            [i for i, j, k, l in zipped_batch],
            [j for i, j, k, l in zipped_batch],
            [k for i, j, k, l in zipped_batch],
            [l for i, j, k, l in zipped_batch]
        ]
        minibatch_anchor_names = unzipped_batch[0]
        minibatch_pos_names = unzipped_batch[1]
        minibatch_neg_names = unzipped_batch[2]
        labels_batch = unzipped_batch[3]

        if m < minibatch_size:
            minibatch_size = m
        m -= minibatch_size
        start = end
        end += minibatch_size

        minibatches.append(((minibatch_anchor_names, minibatch_pos_names, minibatch_neg_names), labels_batch))
        
    return minibatches

def load_images_into_array_from_list_of_dirs(image_names):
    '''
    loads the image set in memory
    
    Args ->
        image_names -- python list containing names of the images with
    Return ->
        data -- numpy array of shape (number of examples, height, width, channels)
    '''
    data = []
    
    # loop througn all the images, store them in memory
    for image_name in image_names:
        image = plt.imread(image_name)
        if image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        data.append(image)
    
    # convert data to np arraay
    data = np.stack(data, axis=3)
    data = np.rollaxis(data, 3)
        
    return data

# unpack csifr file
def unpickle(file):
    '''
    Loads a CSIFR data file into memory
    
    Args ->
        file -- string, file name
    Return ->
        dict -- dictonary having key as class and value as images
    '''
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def check_image_dims(data_dir, f=False):
    '''
    counts number of images in the dataset for a perticular shape
    
    Args ->
        data_dir -- directory where the image set is present
        f - bool if true reads the images in the sub dirs also
    Return ->
        count -- dictonary having key as image shape and value as count
    '''
    i = 0
    tree = ['']
    for file_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + file_name):
            i += 1
        elif f:
            tree.append(file_name+'/')
            for image_name in os.listdir(data_dir+file_name+'/'):
                i += 1
    print('Total number of images =', i)
    
    count = defaultdict(lambda: 0)
    i = 1
    for subdir in tree:
        for image_name in os.listdir(data_dir+subdir):
            if os.path.isfile(data_dir + subdir + image_name):
                s = str(i) + 'images scanned'
                print(s, end='\r')
                image = plt.imread(data_dir +subdir + image_name)
                count[image.shape] += 1
                i += 1
    print()
    print('Completed!!!')
    return count

def rotate_all_images_to_make_the_last_dim_same(data_dir, value, f=False):
    '''
    rotares all the images to make their width same so that we can feed them to the network
    
    Args ->
        data_dir -- directory where the image set is present
        value -- width of the image
        f - bool if true reads the images in the sub dirs also
    '''
    i = 0
    tree = ['']
    for file_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + file_name):
            i += 1
        elif f:
            tree.append(file_name+'/')
            for image_name in os.listdir(data_dir+file_name+'/'):
                i += 1
    print('Total number of images =', i)
    
    i = 1
    for subdir in tree:
        for image_name in os.listdir(data_dir+subdir):
            if os.path.isfile(data_dir + subdir + image_name):
                # print(i, data_dir + image_name)
                image = plt.imread(data_dir + subdir + image_name)
                if image.shape[0] == value:
                    op = str(i) + ': converting '+ str(image.shape) + '    to ' + str(tuple([image.shape[1], image.shape[0], image.shape[2]]))
                    print(op, end="\r")
                    image = np.swapaxes(image, 0, 1)
                    plt.imsave(data_dir + subdir + image_name, image)
                i += 1
    print()
    print('Completed!!!')

def load_images_into_dict(data_dir):
    '''
    loads the image set in memory
    
    Args ->
        data_dir -- directory where the image set is present
    Return ->
        data -- dictonary having key as image shape and value as numpy array of shape (number of examples, height, width, channels)
    '''
    i = 0
    for image_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + image_name):
            i += 1
    print('Total number of images =', i)
    
    data = defaultdict(lambda: [])
    count = defaultdict(lambda: 0)
    i = 1
    # loop througn all the images, store them in memory
    for image_name in os.listdir(data_dir):
        image = plt.imread(data_dir + image_name)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        data[image.shape].append(image)
        count[image.shape] += 1
        print(str(i) + ' images loaded successfully.', end='\r')
        i += 1
    
    # convert data to np arraay
    for key in count.keys():
        print('There are', count[key], 'images of shape', key, end='\r')
        data[key] = np.stack(data[key], axis=3)
        data[key] = np.rollaxis(data[key], 3)
    
    # print('Shape =', data.shape)
    
    return data

def load_images_into_array(data_dir, f=False):
    '''
    loads the image set in memory
    
    Args ->
        data_dir -- directory where the image set is present
        f - bool if true reads the images in the sub dirs also
    Return ->
        data -- numpy array of shape (number of examples, height, width, channels)
    '''
    i = 0
    tree = ['']
    for file_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + file_name):
            i += 1
        elif f:
            tree.append(file_name+'/')
            for image_name in os.listdir(data_dir+file_name+'/'):
                i += 1
    print('Total number of images = ', i)
    data = []
    i = 1
    # loop througn all the images, store them in memory
    for subdir in tree:
        for image_name in sorted(os.listdir(data_dir+subdir)):
            if os.path.isfile(data_dir + subdir + image_name):
                image = plt.imread(data_dir + subdir + image_name)
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                data.append(image.astype('float32'))
                print(str(i) + ' images loaded successfully.', end='\r')
                i += 1
    
    # convert data to np arraay
    data = np.stack(data, axis=3)
    data = np.rollaxis(data, 3)
    
    print()
    print('Shape =', data.shape)
    
    return data

def convert_to_RGB(data_dir, f=False):
    '''
    converts all the images present the data_dir to RGB
    
    Args ->
        data_dir -- directory where the image set is present
        f - bool if true reads the images in the sub dirs also
    '''
    i = 0
    tree = ['']
    for file_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + file_name):
            i += 1
        elif f:
            tree.append(file_name+'/')
            for image_name in os.listdir(data_dir+file_name+'/'):
                i += 1
    print('Total number of images =', i)
    
    i = 1
    for subdir in tree:
        for image_name in os.listdir(data_dir+subdir):
            if os.path.isfile(data_dir + subdir + image_name):
                image = plt.imread(data_dir + subdir + image_name)
                if image.shape[2] != 3:
                    s = str(i) + ': converting ' + str(image.shape) + ' to ' + str((image.shape[0], image.shape[1], 3))
                    # print(s, end="\r")
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    print(rgb_image.shape)
                    plt.imsave(data_dir + subdir + image_name, rgb_image)
                i += 1
    print()
    print('Completed!!!')
  
def resize_all(data_dir, dim, f=False, grey=False):
    '''
    resizes all the images present the data_dir to the dim specifed
    
    Args ->
        data_dir -- directory where the image set is present
        dim -- tupple, desired dim of the image
        f - bool if true reads the images in the sub dirs also
        grey - bool, true if the image is a greyscale image
    '''
    i = 0
    tree = ['']
    for file_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + file_name):
            i += 1
        elif f:
            tree.append(file_name+'/')
            for image_name in os.listdir(data_dir+file_name+'/'):
                i += 1
    print('Total number of images =', i)
    
    i = 1
    for subdir in tree:
        for image_name in os.listdir(data_dir+subdir):
            if os.path.isfile(data_dir + subdir + image_name):
                if grey:
                    image = cv2.imread(data_dir + subdir + image_name, 0)
                else:
                    image = cv2.imread(data_dir + subdir + image_name)
                if (image.shape[0], image.shape[1]) != dim:
                    print(str(i) + ': resizing ' + str(image.shape) + ' to ' + str((dim[0], dim[1])), end="\r")
                    image = cv2.resize(image, (dim[1], dim[0]), interpolation = cv2.INTER_NEAREST)
                    # print(image.shape)
                    cv2.imwrite(data_dir + subdir + image_name, image)
                i += 1
    print()
    print('Completed!!!')

def rotate_and_save(data_dir, angle, zoom=1, f=False):
    '''
    resizes all the images present the data_dir to the angle specifed
    
    Args ->
        data_dir -- directory where the image set is present
        angle -- angle by which the image is to be rotates
    '''
    i = 0
    tree = ['']
    for file_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + file_name):
            i += 1
        elif f:
            tree.append(file_name+'/')
            for image_name in os.listdir(data_dir+file_name+'/'):
                i += 1
    print('Total number of images =', i)
    i = 1
    print()
    
    if not os.path.exists(data_dir+ 'rotated/'):
        os.mkdir(data_dir+ 'rotated/')
    
    for subdir in tree:
        for image_name in os.listdir(data_dir+subdir):
            if os.path.isfile(data_dir + subdir + image_name):
                image = plt.imread(data_dir + subdir + image_name)
                rows, cols, channels = image.shape
                center = (cols/2, rows/2)
                M = cv2.getRotationMatrix2D(center, angle, zoom)
                rotated_image = cv2.warpAffine(image, M, (cols,rows))
                plt.imsave(data_dir + 'rotated/' + str(angle) + '_' + image_name, rotated_image)
                print(str(i) + ' images rotated successfully.', end="\r")
                i += 1
    print()
    
def flip_and_save(data_dir, how, f=False):
    '''
    filps all the images present the data_dir vitically
    
    Args ->
        data_dir -- directory where the image set is present
        how -- string, one of 'horizontal', 'virtical' and 'both'
        f - bool if true reads the images in the sub dirs also
    '''
    patern = {'horizontal' : 0, 'virtical' : 1, 'both' : -1}
    
    i = 0
    tree = ['']
    for file_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + file_name):
            i += 1
        elif f:
            tree.append(file_name+'/')
            for image_name in os.listdir(data_dir+file_name+'/'):
                i += 1
    print('Total number of images =', i)
    i = 1
    print()
    
    if not os.path.exists(data_dir+ 'flip/'):
        os.mkdir(data_dir+ 'flip/')
    
    for subdir in tree:
        for image_name in os.listdir(data_dir+subdir):
            if os.path.isfile(data_dir + subdir + image_name):
                image = plt.imread(data_dir + subdir + image_name)
                flipped_image = cv2.flip( image, patern[how])
                plt.imsave(data_dir + 'flip/' + str(how) + '_' + image_name, rgb_image)
                print(str(i) + ' images flipped successfully.', end="\r")
                i += 1
    print() 
    
def load_image_names_into_list(data_dir, f=False):
    '''
    loads the image names into a list
    
    Args ->
        data_dir -- directory where the image set is present
    Return ->
        data -- python list of length (total number of images)
    '''
    i = 0
    tree = ['']
    data = []
    for file_name in os.listdir(data_dir):
        if os.path.isfile(data_dir + file_name):
            i += 1
        elif f:
            tree.append(file_name+'/')
            for image_name in os.listdir(data_dir+file_name+'/'):
                i += 1
    print('Total number of images =', i)
    i = 1
    print()
    
    # loop througn all the images, store them in memory
    for subdir in tree:
        for image_name in sorted(os.listdir(data_dir+subdir)):
            if os.path.isfile(data_dir + subdir + image_name):
                data.append(data_dir + subdir + image_name)
                print(str(i) + ' images loaded successfully.', end='\r')
                i += 1
        
    return data

def downsample(r, hr_dir, lr_dir):
    '''
    '''
    # check if lr images are present already
    if not os.path.isdir(lr_dir):
        os.mkdir(lr_dir)
    
    # loop througn all the hr images, convert hr to lr by a factor r and save the lr image
    for image_name in os.listdir(hr_dir):
        hr_image = plt.imread(hr_dir + image_name)

        # use bicubic interpolation to reside the images
        lr_image = cv2.resize(hr_image, (int(hr_image.shape[1]/r), int(hr_image.shape[0]/r)), 
                              interpolation=cv2.INTER_CUBIC)
        
        # save the image
        plt.imsave(lr_dir + image_name, lr_image)

def W_relu_init(parameter_dim):
    '''
    Weights initialization if you are using relu
    
    Arguments :
        parameter_dim -- sape of the weight
    Return :
        W - generated weight matrix
    '''
    stddev = 2 / np.sqrt(parameter_dim[-2])
    W = tf.random.normal(shape=parameter_dim, mean= 1.0, stddev=stddev, dtype=tf.dtypes.float32)
    return W

def compute_PSNR(Y_pred, Y, max_value=1.0):
    '''
    Computes PSNR between a set of predicted SR images and original HR images
    
    Arguments:
        Y_pred -- numpy array, predicted SR images
        Y -- numpy array, original HR images
    
    Returns:
        PSNR - Signal to noise ratio value
    '''
    mse = np.mean((Y_pred-Y)**2)
    if mse == 0:
        PSNR = 100
        return PSNR
    else:
        PSNR = 20*math.log10(max_value/math.sqrt(mse))
        return PSNR

def resblock(X, parameters, bn=True):
    '''
    Implemants a single resnet block with skip connection between input and 2 
    CONV2D layers along with batch normalization.
    
    Arguments :
        X -- input tensor to the resblock, must contain same number of filters as the parameters
        parameters -- set of parameters to perform CONV2D operation
        bn -- boolean if True, then perform batch normalozation
        
    Return :
        S_bn -- normalized output tensor of the resblock
    '''
    W = []
    for weight, parameter in parameters.items():
        W.append(weight)
    # CONV2D: stride of 1, padding 'SAME'
    Z  = tf.nn.conv2d(X, parameters[W[0]], strides=[1, 1, 1, 1], padding='SAME', name='CONV2D_res'+weight)
    # RELU
    A = tf.nn.relu(Z, name='relu_res'+weight)
    # CONV2D: stride of 1, padding 'SAME'
    Z  = tf.nn.conv2d(X, parameters[W[1]], strides=[1, 1, 1, 1], padding='SAME', name='CONV2D_res'+weight)
    
    if bn:
        # batch normalization
        X_mean, X_var = tf.nn.moments(X, [0])
        A_mean, A_var = tf.nn.moments(A, [0])
        X_bn = tf.nn.batch_normalization(X, X_mean, X_var, offset=0, scale=1, variance_epsilon=0.001)
        A_bn = tf.nn.batch_normalization(A, A_mean, A_var, offset=0, scale=1, variance_epsilon=0.001)

        # feed forward
        S = tf.add(X, A)

        # batch normalization
        S_mean, S_var = tf.nn.moments(S, [0])
        S_bn = tf.nn.batch_normalization(S, S_mean,S_var, offset=0, scale=1, variance_epsilon=0.001)

        return S_bn
    else:
        S = tf.add(X, A)
        return S