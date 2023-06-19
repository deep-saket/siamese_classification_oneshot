from email.mime import image
from builtins import len, print
from cProfile import label
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import OrderedDict
import math
import os
import random
import pista
from scipy import ndimage
import random
from pprint import pprint

def normalize(data_batch):
    '''
    Normalizes the inopt batch
    '''
    norm_batch = []

    norm_batch.append(data_batch[0] / 255.0)
    norm_batch.append(data_batch[-1] / 255.0)

    return tuple(norm_batch)
class TripletLoader:
    '''
    Loads the datset.
    Returns as
    [
        (
            ndarray of share BATCH_SIZExHxWxC, 
            ndarray of share BATCH_SIZExHxWxC, 
            ndarray of share BATCH_SIZExHxWxC
        )
        ndarray of shape 1xBATCH_SIZE
    ]
    '''
    def __init__(self, batch_size, dataset_path, grayscale=False, hflip=False, vflip=False, rotate=False, blur=False, size=(224, 224), visualize=False):
        '''
        Initialize the class.
        Args ::
            batch_size -- int | number of examples to be processed in one example
            dataset_path -- str | tfrecord dir
            dataset_info_path -- str | info.txt path
        '''
        self._batch_size = batch_size
        self._dataset_path = dataset_path
        self._fetched_batch = 0
        self.grayscale = grayscale
        self.rotate = rotate
        self.blur = blur
        self.hflip = hflip
        self.vflip = vflip
        self.size = size
        self.visualize = visualize

       ## get number of classes present in the dataset
        self.classes = os.listdir(dataset_path)
        self.n_classes = len(self.classes)
        self.class_number_dict = {self.classes[i] : i for i in range(len(self.classes))}

        ## class dir dict
        self.class_dict = {klass : os.path.join(dataset_path, klass) for klass in self.classes}
        
        ## prepare the dataset
        self.dataset = self._prepare_data()
        random.shuffle(self.dataset)
                
    def _prepare_data(self):
        '''
        Loads examples into memory from tfrecord.
        '''
        anchor_image_paths = []
        pos_image_paths = []
        neg_image_paths = []
        lables = []

        m = 0
        for klass_name, klass_path in self.class_dict.items():
            m += 1
            file_names = os.listdir(klass_path)

            i = 0
            for file_name in file_names:
                print(f'{i}, {m} /  {len(file_names)}, {len(self.class_dict)}', end='\r')
                i += 1
                if file_name.split('.')[-1] not in ['jpg', 'jpeg', 'JPG', 'png', 'PNG', 'tiff', 'tif', 'TIFF', 'TIF']:
                    continue
                
                anchor_image_path = os.path.join(klass_path, file_name)

                remaining_pos = list(file_names)
                remaining_pos.remove(file_name)
                random.shuffle(remaining_pos)
                pos_image_path = os.path.join(klass_path, remaining_pos[0])

                remaining_klass = list(self.classes)
                remaining_klass.remove(klass_name)
                random.shuffle(remaining_klass)
                remaining_klass_image_paths = os.listdir(self.class_dict[remaining_klass[0]])
                random.shuffle(remaining_klass_image_paths)
                neg_image_path = os.path.join(self.class_dict[remaining_klass[0]], remaining_klass_image_paths[0])
            
                pos_image_paths.append(pos_image_path)
                anchor_image_paths.append(anchor_image_path)
                neg_image_paths.append(neg_image_path)

                lables.append(self.class_number_dict[klass_name])
        
        self.m = m
        return pista.random_mini_batches_from_SSD_tripple(anchor_image_paths, pos_image_paths, neg_image_paths, lables, self._batch_size)

    def show_triplet(self, pairs):
        anchor, positive, negative = pairs
        fig, ax = plt.subplots(10,3, figsize=(10,20))
        
        for i in range(10):
            ax[i, 0].imshow((anchor[i]))
            ax[i, 1].imshow((positive[i]))
            ax[i, 2].imshow((negative[i]))
            ax[i, 0].set_title("anchor")
            ax[i, 1].set_title("Positive")
            ax[i, 2].set_title("Negative")
            ax[i, 0].axis('off')
            ax[i, 1].axis('off')
            ax[i, 2].axis('off')
        
    def _get_items(self, image_paths, labels):
        ancchor_images = []
        pos_images = []
        neg_images = []

        for i in range(len(labels)):
            ancchor_image_path = image_paths[0][i]
            pos_image_path = image_paths[1][i]
            neg_image_path  = image_paths[2][i]
            # print(2, image_paths[i])
            ancchor_image, pos_image, neg_image = self._get_item((ancchor_image_path, pos_image_path, neg_image_path))
            # if self.visualize:
            #    self.show_triplet((ancchor_image, pos_image, neg_image))
            ancchor_images.append(ancchor_image)
            pos_images.append(pos_image)
            neg_images.append(neg_image)
            
        ancchor_images = np.stack(ancchor_images, axis=3)
        ancchor_images = np.rollaxis(ancchor_images, 3)
        
        pos_images = np.stack(pos_images, axis=0)
        pos_images = np.rollaxis(pos_images, axis=0)

        neg_images = np.stack(neg_images, axis=0)
        neg_images = np.rollaxis(neg_images, axis=0)
        
        return (ancchor_images, pos_images, neg_images), np.array(labels).reshape((1, len(labels)))
            
         
    def _get_item(self, image_path):      
        ancchor_image_path, pos_image_path, neg_image_path = image_path
        ancchor_image = plt.imread(ancchor_image_path)
        pos_image = plt.imread(pos_image_path)
        neg_image = plt.imread(neg_image_path)

        h, w, _ = ancchor_image.shape
        
        if ancchor_image.shape[-1] > 3:
            ancchor_image = cv2.cvtColor(ancchor_image, cv2.COLOR_RGBA2RGB)
        if pos_image.shape[-1] > 3:
            pos_image = cv2.cvtColor(pos_image, cv2.COLOR_RGBA2RGB)        
        if neg_image.shape[-1] > 3:
            neg_image = cv2.cvtColor(neg_image, cv2.COLOR_RGBA2RGB)        
        
        if self.grayscale:
            ancchor_image = cv2.cvtColor(ancchor_image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
            pos_image = cv2.cvtColor(pos_image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
            neg_image = cv2.cvtColor(neg_image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        
        ancchor_image = tf.image.resize(ancchor_image, self.size)
        pos_image = tf.image.resize(pos_image, self.size)
        neg_image = tf.image.resize(neg_image, self.size)

        if self.rotate:
            if random.random <= 0.5:
                ancchor_image = tf.image.rot90(ancchor_image)
                pos_image = tf.image.rot90(pos_image)
                neg_image = tf.image.rot90(neg_image)

        if self.hflip:
            ancchor_image = tf.image.random_flip_left_right(ancchor_image)
            pos_image = tf.image.random_flip_left_right(pos_image)
            neg_image =tf.image.random_flip_left_right(neg_image)
        if self.vflip:
            ancchor_image = tf.image.random_flip_up_down(ancchor_image)
            pos_image = tf.image.random_flip_up_down(pos_image)
            neg_image = tf.image.random_flip_up_down(neg_image)
        
        if (ancchor_image_path.split('.')[-1] != 'png' and ancchor_image_path.split('.')[-1] != 'PNG') or ancchor_image.numpy().max() > 2.:
            ancchor_image = (ancchor_image / 255.)
        if (pos_image_path.split('.')[-1] != 'png' and pos_image_path.split('.')[-1] != 'PNG') or pos_image.numpy().max() > 2.:
            pos_image = (pos_image / 255.)
        if (neg_image_path.split('.')[-1] != 'png' and neg_image_path.split('.')[-1] != 'PNG') or neg_image.numpy().max() > 2.:
            neg_image = (neg_image / 255.)
    
        return ancchor_image, pos_image, neg_image
    
    def get_data(self):
        '''
        fetch one batch at a time
        '''
        if self._fetched_batch >= len(self.dataset):
            self.dataset = self._prepare_data()
            random.shuffle(self.dataset)
            self._fetched_batch = 0

        ## fetch 1 batch at a time from the iterator
        data_batch_paths = self.dataset[self._fetched_batch]
        self._fetched_batch += 1
        
        image_paths, labels = data_batch_paths
        images, labels = self._get_items(image_paths, labels)
        data_batch = {'images': images, 'labels': labels}

        return data_batch
    
    def count_minibatches(self):
        '''
        Returns number of minibatches.
        '''
        return len(self.dataset)