import tensorflow as tf

def comCNN(orig_img,is_training=False, l2_regularizer=True,reuse=False):
    # add regularization 
    if l2_regularizer:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    else :
        regularizer = None
    with tf.variable_scope("comCNN",reuse=reuse) as vs: 
        # layer 1
        conv1 = tf.layers.conv2d(
        inputs=orig_img,
        filters=64,
        kernel_size=[3, 3],
        activation=tf.nn.relu,
        padding='SAME',
        kernel_regularizer=regularizer,
        name='com/conv1')
        
        # conv1 = tf.Print(conv1,[conv1],"After first convolution: ",summarize=20)

        # layer 2 : 
        conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        strides=(2,2),
        kernel_size=[3, 3],
        padding='SAME',
        kernel_regularizer=regularizer,
        name='com/conv2')
        
        conv2 = tf.layers.batch_normalization(conv2,training=is_training,
        name='com/bn2')
        conv2 = tf.nn.relu(conv2, name='com/a2')

        # conv2 = tf.Print(conv2,[conv2],"After second convolution: ",summarize=20)

        #layer 3 :
        conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=1,
        kernel_size=[3, 3],
        kernel_regularizer=regularizer,
        name='com/conv3')

        # conv3 = tf.Print(conv3,[conv3],"After third convolution: ",summarize=20)       

        return conv3


def recCNN(orig_img,img,no_of_layers,is_training=False,l2_regularizer=True,reuse=False):
    # add regularization 
    if l2_regularizer:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    else :
        regularizer = None 

    with tf.variable_scope("comCNN",reuse=reuse) as vs: 

        #bicubic interpolation
        upscaled_img =  tf.image.resize_bicubic(img, [200,200],name='rec/bicubic1')

        #layer 1
        conv1 = tf.layers.conv2d(
        inputs=upscaled_img,
        filters=64,
        kernel_size=[3, 3],
        activation=tf.nn.relu,
        padding='SAME',
        kernel_regularizer=regularizer,
        name='rec/conv1')
        #layer 2 to 19
        conv2 = conv1

        for i in range(no_of_layers-2):
            conv2 = tf.layers.conv2d(
                inputs=conv2,
                filters=64,
                kernel_size=[3, 3],
                padding='SAME',
                kernel_regularizer=regularizer,
                name='rec/conv/%s'%(i+2))
            conv2 = tf.layers.batch_normalization(conv2,training=is_training,name='rec/bn/%s'%(i+2))
            conv2 = tf.nn.relu(conv2,name='rec/a/%s'%(i+2))

        #layer 20
        residual_img = tf.layers.conv2d(
            inputs=conv2,
            filters=1,
            kernel_size=[3, 3],
            padding='SAME',
            kernel_regularizer=regularizer,
            name='rec/conv_final')
        
        #residual learning
        # final_img = residual_img
        final_img = tf.math.add(upscaled_img,residual_img,
            name='rec/add_final')

        return (final_img,residual_img,upscaled_img)
