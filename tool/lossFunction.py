import numpy as np
import tensorflow as tf

def discount_reward_crossentropy(y_true, y_pred):
    neg_log_pr0b = tf.nn.sparse_softmax_cross_entropy_with_logits()
