import numpy as np
import tensorflow as tf
from time import time
import sys
import time
from PIL import Image

import end_to_end_net
import utils

L2_REGULARIZER = True
NUM_LAYERS_IN_recCNN = 20
NUM_EPOCH = 1
BATCH_SIZE = 128
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
IMAGE_DEPTH = 3
TRAINING = False


orig_image = utils.train_input_fn(utils.read_all_images("./data/airplanes/*.jpg",IMAGE_HEIGHT,IMAGE_WIDTH), BATCH_SIZE)

conv_img =  end_to_end_net.comCNN(orig_image,L2_REGULARIZER)

(final_img,residual_img,upscaled_img)  = end_to_end_net.recCNN(orig_image,conv_img,7,L2_REGULARIZER)
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                        1, 0.96, staircase=True)
loss_1 = tf.losses.mean_squared_error(orig_image,final_img)

loss_2 = tf.losses.mean_squared_error(residual_img , (upscaled_img - orig_image))

optimizer1 = (tf.train.AdamOptimizer(learning_rate).minimize(loss_1, global_step=global_step))
optimizer2 = (tf.train.AdamOptimizer(learning_rate).minimize(loss_2, global_step=global_step))

sess = tf.Session()
saver = tf.train.Saver()

if TRAINING: 
    init = tf.global_variables_initializer()
    sess.run(init)
    f = open("log.txt","w")
    for i in range(200):
        start_time = time.time()
        print "iteration: "+str(i)
        sess.run(optimizer1)
        loss1 = sess.run(loss_1)
        print "loss 1 :"+str(loss1)
        sess.run(optimizer2)
        loss2 = sess.run(loss_2)
        print "loss 2 :"+str(loss2)
        f.write("Iteration :"+str(i)+" loss1 : "+str(loss1)+" loss2 :"+str(loss2)+"\n"+"time : "+str(time.time() - start_time)+"\n" )
    saver = tf.train.Saver()
    saver.save(sess,'./model.ckpt')
else :
    img_path = "./data/Faces_easy/image_0001.jpg"
    orig_image = utils.read_image(img_path,IMAGE_HEIGHT,IMAGE_WIDTH)
    orig_image = tf.expand_dims(orig_image, 0)
    conv_img =  end_to_end_net.comCNN(orig_image,L2_REGULARIZER)
    (final_img,residual_img,upscaled_img)  = end_to_end_net.recCNN(orig_image,conv_img,7,L2_REGULARIZER)
    saver.restore(sess, "./model.ckpt")  
    print("Model restored.")
    init = tf.global_variables_initializer()
    sess.run(init)
    utils.resize_img(img_path,IMAGE_HEIGHT,IMAGE_WIDTH)
    conv_img = sess.run(tf.squeeze(conv_img))
    utils.write_jpeg(conv_img,"./mid_im.jpg")
    final_im = sess.run(tf.squeeze(final_img))
    # print type(final_im)
    # print final_im
    utils.write_jpeg(final_im,"./final_im.jpg")
    # HEIC 