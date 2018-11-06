 # -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from time import time
import sys
import time
from PIL import Image
from keras.datasets import cifar10

import end_to_end_net
import utils

L2_REGULARIZER = False
NUM_LAYERS_IN_recCNN = 20
NUM_EPOCH = 2
BATCH_SIZE = 16
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
IMAGE_DEPTH = 3
TRAINING = True

#/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation

def training():
    input_img = utils.train_input_fn(utils.read_all_images("./data/airplanes/*.jpg",IMAGE_HEIGHT,IMAGE_WIDTH), BATCH_SIZE)
    conv_img =  end_to_end_net.comCNN(input_img,True,L2_REGULARIZER,reuse=False)
    (final_img,residual_img,upscaled_img)  = end_to_end_net.recCNN(input_img,conv_img,NUM_LAYERS_IN_recCNN,
            True,L2_REGULARIZER,reuse=False)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                        1, 0.96, staircase=True)


    tf.summary.scalar('learning_rate', learning_rate)

    loss_1 = tf.losses.mean_squared_error(input_img,final_img)

    loss_2 = tf.losses.mean_squared_error(residual_img , (upscaled_img - input_img))

    tf.summary.scalar('loss_1', loss_1)
    tf.summary.scalar('loss_2', loss_2)


    optimizer1 = (tf.train.AdamOptimizer(learning_rate).minimize(loss_1, global_step=global_step))
    optimizer2 = (tf.train.AdamOptimizer(learning_rate).minimize(loss_2, global_step=global_step))

    sess = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    # f = open("./log.txt","w")

    for var in tf.trainable_variables():
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)        
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train', graph=tf.get_default_graph())

    for i in range(NUM_EPOCH):
        start_time = time.time()
        print "iteration: "+str(i)
        sess.run(optimizer1)
        # loss1 = sess.run(loss_1)
        # print "loss 1 :"+str(loss1)
        sess.run(optimizer2)
        summary = sess.run(merged_summary)
        train_writer.add_summary(summary, i)
        # loss2 = sess.run(loss_2)
        # print "loss 2 :"+str(loss2)
        # f.write("Iteration :"+str(i)+" loss1 : "+str(loss1)+" loss2 :"+str(loss2)+"\n"+"time : "+str(time.time() - start_time)+"\n" )

    saver.save(sess,'./model.ckpt')



def inference():
    img_path = "/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation/data/airplanes/image_0002.jpg"
    orig_image = utils.read_image(img_path,IMAGE_HEIGHT,IMAGE_WIDTH)
    # print "image before expand_dimentions"
    # print orig_image
    orig_image = tf.expand_dims(orig_image, 0)
    # print "after expand_dim"
    # print orig_image
    conv_img =  end_to_end_net.comCNN(orig_image,False,L2_REGULARIZER,reuse=tf.AUTO_REUSE)
    (final_img, residual_img, upscaled_img)  = end_to_end_net.recCNN(orig_image,conv_img,NUM_LAYERS_IN_recCNN,False,L2_REGULARIZER,reuse=tf.AUTO_REUSE)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, "/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation/model.ckpt")  
    init = tf.global_variables_initializer()
    sess.run(init)
    utils.resize_img(img_path,IMAGE_HEIGHT,IMAGE_WIDTH)
    conv_img = sess.run(tf.squeeze(conv_img,[0]))
    utils.write_jpeg(conv_img,"/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation/mid_im.jpg")
    final_im = sess.run(tf.squeeze(final_img,[0]))
    utils.write_jpeg(final_im,"/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation/final_im.jpg")

if (sys.argv[1]).upper() == "TRAIN" :
    TRAINING = True
else:
    TRAINING = False

if TRAINING: 
    training()
else :
    inference()