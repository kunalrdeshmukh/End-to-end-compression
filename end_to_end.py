import numpy as np
import tensorflow as tf
from time import time
import sys
import time
from PIL import Image

import end_to_end_net
import utils

L2_REGULARIZER = False
NUM_LAYERS_IN_recCNN = 20
NUM_EPOCH = 50
BATCH_SIZE = 16
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
IMAGE_DEPTH = 3
TRAINING = True

#/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation

def training():
    input_img = utils.train_input_fn(utils.read_all_images("./data/airplanes/*.jpg",IMAGE_HEIGHT,IMAGE_WIDTH), BATCH_SIZE)
    print input_img
    conv_img =  end_to_end_net.comCNN(input_img,True,L2_REGULARIZER,reuse=False)
    (final_img,residual_img,upscaled_img)  = end_to_end_net.recCNN(input_img,conv_img,NUM_LAYERS_IN_recCNN,
            True,L2_REGULARIZER,reuse=False)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                        1, 0.96, staircase=True)
    loss_1 = tf.losses.mean_squared_error(input_img,final_img)

    loss_2 = tf.losses.mean_squared_error(residual_img , (upscaled_img - input_img))

    optimizer1 = (tf.train.AdamOptimizer(learning_rate).minimize(loss_1, global_step=global_step))
    optimizer2 = (tf.train.AdamOptimizer(learning_rate).minimize(loss_2, global_step=global_step))

    sess = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    f = open("log.txt","w")
    for i in range(NUM_EPOCH):
        start_time = time.time()
        print "iteration: "+str(i)
        sess.run(optimizer1)
        loss1 = sess.run(loss_1)
        print "loss 1 :"+str(loss1)
        sess.run(optimizer2)
        loss2 = sess.run(loss_2)
        print "loss 2 :"+str(loss2)
        f.write("Iteration :"+str(i)+" loss1 : "+str(loss1)+" loss2 :"+str(loss2)+"\n"+"time : "+str(time.time() - start_time)+"\n" )
    saver.save(sess,'./model.ckpt')



def inference():
    img_path = "/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation/data/airplanes/image_0002.jpg"
    orig_image = utils.read_image(img_path,IMAGE_HEIGHT,IMAGE_WIDTH)
    orig_image = tf.expand_dims(orig_image, 0)
    conv_img =  end_to_end_net.comCNN(orig_image,False,L2_REGULARIZER,reuse=tf.AUTO_REUSE)
    (final_img,residual_img,upscaled_img)  = end_to_end_net.recCNN(orig_image,conv_img,NUM_LAYERS_IN_recCNN,False,L2_REGULARIZER,reuse=tf.AUTO_REUSE)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, "/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation/model.ckpt")  
    print("Model restored.")

    init = tf.global_variables_initializer()
    sess.run(init)
    utils.resize_img(img_path,IMAGE_HEIGHT,IMAGE_WIDTH)
    conv_img = sess.run(tf.squeeze(conv_img))
    utils.write_jpeg(conv_img,"/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation/mid_im.jpg")
    final_im = sess.run(tf.squeeze(final_img))
    utils.write_jpeg(final_im,"/content/drive/My Drive/Fall 18/297/Deliverable_1/End_to_End_implementation/final_im.jpg")

if (sys.argv[1]).upper() == "TRAIN" :
    TRAINING = True
else:
    TRAINING = False

if TRAINING: 
    training()
else :
    inference()
