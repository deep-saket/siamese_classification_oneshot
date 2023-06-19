import tensorflow as tf

def triplet_cost(Y, Y_pred, margin=0.5):
    '''
    Computes triplet cost
    '''
    # The output of the network is a tuple containing the distances
    # between the anchor and the positive example, and the anchor and
    # the negative example.
    def _contrastive_loss(y_true, y_pred):
        return tfa.losses.contrastive_loss(y_true, y_pred)
        
    loss = tf.convert_to_tensor(0,dtype=tf.float32)
    g = tf.constant(1.0, shape=[1], dtype=tf.float32)
    h = tf.constant(0.0, shape=[1], dtype=tf.float32)
    
    ap_distance, an_distance = Y_pred
    # print(ap_distance, an_distance)
    # loss_query_pos = tf.reduce_mean(_contrastive_loss(g, ap_distance))
    # loss_query_neg = tf.reduce_mean(_contrastive_loss(h, an_distance))
    loss = ap_distance - an_distance
    
    # Computing the Triplet Loss by subtracting both distances and
    # making sure we don't get a negative value.
    loss = tf.maximum(loss + margin, 0.0)
    
    return loss