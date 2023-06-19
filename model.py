import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16, mobilenet_v3, MobileNetV3Small
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import numpy as np
import keras_toolkit as kt

def b_init(shape, dtype=tf.float32, name=None):
        """Initialize bias as in paper"""
        values = np.random.normal(loc=0.5,scale=1e-2,size=shape)
        return K.variable(values,name=name)
    
def W_init(shape, dtype=tf.float32, name=None):
        """Initialize weights as in paper"""
        values = np.random.normal(loc=0,scale=1e-2,size=shape)
        return K.variable(values,name=name)

def get_layers_output_by_name(model, layer_names):
    '''
    Retrives layers from a model given the model and layer names.
    '''
    return {v : model.get_layer(v).output for v in layer_names}

strategy = kt.accelerator.auto_select(verbose=True)

def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values = np.random.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

class VGG16Helper:
    @classmethod
    def get_embeddings(cls, target_shape = (224, 224)):
        '''
        This function returns vgg embeddings.
        '''
        with strategy.scope():
            ## create vgg model and set trainable false
            vgg_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=target_shape + (3,))
            for layer in vgg_model.layers[:10]:
                    layer.trainable = False

            ## extract intermediate outputs
            intermediate_layer_outputs = get_layers_output_by_name(vgg_model, 
                                                                ["block1_pool", "block2_pool", "block3_pool", "block4_pool"])

            convnet_output = layers.GlobalAveragePooling2D()(vgg_model.output)
            for layer_name, output in intermediate_layer_outputs.items():
                    output = layers.GlobalAveragePooling2D()(output)
                    convnet_output = layers.concatenate([convnet_output, output])

            convnet_output = layers.Dense(512, activation = 'relu')(convnet_output)
            convnet_output = layers.Dropout(0.6)(convnet_output)
            convnet_output = layers.Dense(512, activation = 'relu')(convnet_output)
            convnet_output = layers.Lambda(lambda p: K.l2_normalize(p,axis=1))(convnet_output)
            
            return Model(inputs=[vgg_model.input], outputs=convnet_output,name="Embedding")
    
    @classmethod
    def preprocess_input(cls, input):
        return vgg16.preprocess_input(input)

class MobileNetV3Helper:
    @classmethod
    def get_embeddings(cls, target_shape = (224, 224), trainable=False):
        '''
        This function returns vgg embeddings.
        '''
        with strategy.scope():
            ## create vgg model and set trainable false
            model = MobileNetV3Small(weights="imagenet", include_top=False, input_shape=target_shape + (3,))
            for layer in model.layers[:-50]:
                layer.trainable = trainable

            ## extract intermediate outputs
            intermediate_layer_outputs = get_layers_output_by_name(model, 
                                                                ['expanded_conv_4/squeeze_excite/AvgPool', 
                                                                'expanded_conv_6/squeeze_excite/AvgPool', 
                                                                'expanded_conv_8/squeeze_excite/AvgPool', 
                                                                'expanded_conv_10/squeeze_excite/AvgPool'])

            convnet_output = layers.GlobalAveragePooling2D()(model.output)
            for layer_name, output in intermediate_layer_outputs.items():
                    output = layers.GlobalAveragePooling2D()(output)
                    convnet_output = layers.concatenate([convnet_output, output])

            convnet_output = layers.Dense(512, activation = 'relu')(convnet_output)
            convnet_output = layers.Dropout(0.6)(convnet_output)
            convnet_output = layers.Dense(512, activation = 'relu')(convnet_output)
            convnet_output = layers.Lambda(lambda p: K.l2_normalize(p,axis=1))(convnet_output)
            
            return Model(inputs=[model.input], outputs=convnet_output,name="Embedding")

    @classmethod
    def preprocess_input(cls, input):
        return mobilenet_v3.preprocess_input(input)



class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_mean(tf.reduce_sum(tf.square(anchor - positive), -1))
        an_distance = tf.reduce_mean(tf.reduce_sum(tf.square(anchor - negative), -1))
        return (ap_distance, an_distance)


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The Contrastive Loss is defined as:
       L(θ) = (1-y)(1/2)D(Xq, Xp)^2 + y(1/2){max(0, m - D(Xq, Xn)^2)}
    """

    def __init__(self, target_shape = (224, 224), EmbeddingHelper=VGG16Helper, training=True):
        super(SiameseModel, self).__init__()

        self.target_shape = target_shape
        self.EmbeddingHelper = EmbeddingHelper
        self.embeddings = self.EmbeddingHelper.get_embeddings(target_shape = self.target_shape)
        self.siamese_network = self._create_siamese_net()

        self.training = training

    def _create_siamese_net(self):
        '''
        This Function creates siamese network.
        '''
        with strategy.scope():
            anchor_input = layers.Input(name="anchor", shape=self.target_shape + (3,))
            positive_input = layers.Input(name="positive", shape=self.target_shape + (3,))
            negative_input = layers.Input(name="negative", shape=self.target_shape + (3,))

            distances = DistanceLayer()(
                self.embeddings(self.EmbeddingHelper.preprocess_input(anchor_input)),
                self.embeddings(self.EmbeddingHelper.preprocess_input(positive_input)),
                self.embeddings(self.EmbeddingHelper.preprocess_input(negative_input)),
            )

            siamese_network = Model(
                inputs=[anchor_input, positive_input, negative_input], outputs=distances
            )
        return siamese_network

    def call(self, inputs):
        if self.training:
            return self.siamese_network(inputs)
        else:
            return self.embeddings(inputs)

class SiameseModelOneShot(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The Contrastive Loss is defined as:
       L(θ) = (1-y)(1/2)D(Xq, Xp)^2 + y(1/2){max(0, m - D(Xq, Xn)^2)}
    """

    def __init__(self, target_shape = (224, 224), EmbeddingHelper=MobileNetV3Helper, training=True):
        super(SiameseModelOneShot, self).__init__()

        self.target_shape = target_shape
        self.EmbeddingHelper = EmbeddingHelper
        self.embeddings = self.EmbeddingHelper.get_embeddings(target_shape = self.target_shape)
        self.siamese_network = self._create_siamese_net()

        self.training = training
        

    def _create_siamese_net(self):
        '''
        This Function creates siamese network.
        '''
        

        #//TODO: figure out how to initialize layer biases in keras.
        left_input = layers.Input(name="anchor", shape=self.target_shape + (3,))
        right_input = layers.Input(name="positive", shape=self.target_shape + (3,))

        encoded_l = MobileNetV3Helper.get_embeddings(left_input)
        encoded_r = MobileNetV3Helper.get_embeddings(right_input)        

        #merge two encoded inputs with the l1 distance between them
        subtracted = layers.Subtract()( [encoded_l,encoded_r]  )
        both = layers.Lambda(lambda x: abs(x))(subtracted)
        prediction = layers.Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
        siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

        return siamese_net

    def call(self, inputs):
        if self.training:
            return self.siamese_network(inputs)
        else:
            return self.embeddings(inputs)