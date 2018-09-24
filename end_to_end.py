import numpy as np
import tensorflow as tf
from time import time
import sys

import end_to_end_net
import utils

L2_REGULARIZER = False
NUM_LAYERS_IN_recCNN = 20
NUM_EPOCH = 50
BATCH_SIZE = 128
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
IMAGE_DEPTH = 3

# print end_to_end_net.recCNN(utils.read_all_images("./data/airplanes/*.jpg",180,180),end_to_end_net.comCNN(utils.read_all_images("./data/airplanes/*.jpg",180,180),False),20,False)


# def get_feature_columns():
#   feature_columns = {
#     'images': tf.feature_column.numeric_column('images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
#   }
#   return feature_columns
# feature_columns = get_feature_columns()

orig_image = utils.train_input_fn(utils.read_all_images1("./data/airplanes/*.jpg",180,180), BATCH_SIZE)

conv_img =  end_to_end_net.comCNN(orig_image,True)

(final_img,residual_img,upscaled_img)  = end_to_end_net.recCNN(orig_image,conv_img,20,True)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                        50, 0.96, staircase=True)
loss_1 = tf.losses.mean_squared_error(orig_image,final_img)

loss_2 = tf.losses.mean_squared_error(residual_img , (upscaled_img - orig_image))

optimizer1 = (tf.train.AdamOptimizer(learning_rate).minimize(loss_1, global_step=global_step))
optimizer2 = (tf.train.AdamOptimizer(learning_rate).minimize(loss_2, global_step=global_step))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(optimizer1)
    sess.run(optimizer2)
        
