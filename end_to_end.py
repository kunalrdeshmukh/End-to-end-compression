import numpy as np
import tensorflow as tf

import end_to_end_net
import utils


print end_to_end_net.recCNN(utils.read_all_images("./data/airplanes/*.jpg",180,180),end_to_end_net.comCNN(utils.read_all_images("./data/airplanes/*.jpg",180,180),False),20,False)

def train():

    # rate = tf.train.exponential_decay(0.001, batch * BATCH_SIZE, train_size, 0.9999)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           50, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
    # loss1 = tf.losses.mean_squared_error(orig_img,final_img)
    # loss2 = tf.losses.mean_squared_error(residual_img - (upscaled_img - orig_img))

    # learning_step1 = (tf.train.AdamOptimizer(learning_rate).minimize(loss1, global_step=global_step))
    # learning_step2 = (tf.train.AdamOptimizer(learning_rate).minimize(loss2, global_step=global_step))