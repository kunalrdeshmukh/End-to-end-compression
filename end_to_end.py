import cv2
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

# def read_images(path,x,y):
#     data = []
#     files = glob.glob (path)
#     for myFile in files:
#         image = cv2.imread(myFile)
#         resized = cv2.resize(image,(x,y))
#         data.append (resized)
#     return data

def read_images_tf(path,x,y):
    image = tf.image.decode_jpeg(path,channels=3)
    resized_image = tf.expand_dims(tf.image.resize_images(image, [x, y]),0)
    return resized_image


def comCNN(img):

    # layer 1
    conv1 = tf.layers.conv2d(
      inputs=img,
      filters=64,
      kernel_size=[3, 3],
      activation=tf.nn.relu,
      padding='SAME')

    # layer 2 : 
    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      strides=(2,2),
      kernel_size=[3, 3],
      padding='SAME')
    
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.relu(conv2)

    #layer 3 :
    conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=3,
      kernel_size=[3, 3])

    return conv3


def recCNN(img,no_of_layers):

    #bicubic interpolation
    img =  tf.image.resize_bicubic(img, [180,180])

    #layer 1
    conv1 = tf.layers.conv2d(
      inputs=img,
      filters=64,
      kernel_size=[3, 3],
      activation=tf.nn.relu,
      padding='SAME')
    #layer 2 to 19
    conv2 = conv1
    for _ in range(no_of_layers-2):
        conv2 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME')
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)

    #layer 20
    conv20 = tf.layers.conv2d(
        inputs=conv2,
        filters=3,
        kernel_size=[3, 3],
        padding='SAME')
    
    #residual learning
    final_img = tf.add(img,conv20)

    return final_img


# data = read_images("./data/airplanes/*.jpg",180,180)
files = glob.glob ("./data/airplanes/*.jpg")
for myFile in files:
    image = read_images_tf(myFile,180,180)
    im =  comCNN(image)
    recon_im = recCNN(im,20)