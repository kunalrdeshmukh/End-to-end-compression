import tensorflow as tf

def comCNN(orig_img,l2_regularizer):

    # add regularization 
    if l2_regularizer:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    else :
        regularizer = None 
    # layer 1
    conv1 = tf.layers.conv2d(
      inputs=orig_img,
      filters=64,
      kernel_size=[3, 3],
      activation=tf.nn.relu,
      padding='SAME',
      kernel_regularizer=regularizer)

    # layer 2 : 
    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      strides=(2,2),
      kernel_size=[3, 3],
      padding='SAME',
      kernel_regularizer=regularizer)
    
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.relu(conv2)

    #layer 3 :
    conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=3,
      kernel_size=[3, 3],
      kernel_regularizer=regularizer)

    return conv3


def recCNN(orig_img,img,no_of_layers,l2_regularizer):

    # add regularization 
    if l2_regularizer:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    else :
        regularizer = None 

    #bicubic interpolation
    upscaled_img =  tf.image.resize_bicubic(img, [180,180])

    #layer 1
    conv1 = tf.layers.conv2d(
      inputs=upscaled_img,
      filters=64,
      kernel_size=[3, 3],
      activation=tf.nn.relu,
      padding='SAME',
      kernel_regularizer=regularizer)
    #layer 2 to 19
    conv2 = conv1
    for _ in range(no_of_layers-2):
        conv2 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            kernel_regularizer=regularizer)
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)

    #layer 20
    residual_img = tf.layers.conv2d(
        inputs=conv2,
        filters=3,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_regularizer=regularizer)
    
    #residual learning
    final_img = tf.add(orig_img,residual_img)

    return (final_img,residual_img,upscaled_img)
