import glob
import tensorflow as tf

def read_all_images(path,x,y):
    files = glob.glob (path)
    data = []
    for myFile in files:
        #print myFile
        value = tf.read_file(myFile)
        image = tf.image.decode_jpeg(value,channels=3)
        resized_image = tf.image.resize_images(image, [x, y])
        data.append(resized_image)
    return tf.stack(data)

def read_image(path,x,y):
    files = glob.glob (path)
    value = tf.read_file(files[0])
    image = tf.image.decode_jpeg(value,channels=3)
    resized_image = tf.image.resize_images(image, [x, y])
    return resized_image


def write_jpeg(data, filepath):
    g = tf.Graph()
    with g.as_default():
        data_t = tf.placeholder(tf.uint8)
        op = tf.image.encode_jpeg(data_t, format='rgb', quality=100)
        init = tf.initialize_all_variables()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        data_np = sess.run(op, feed_dict={ data_t: data })

    with open(filepath, 'w') as fd:
        fd.write(data_np)
        
def write_img(data,filepath):
    img = tf.image.encode_jpeg(data)
    tf.write_file(filepath,img)

def train_input_fn(features, batch_size):
    """ Input function for training """
    #convert input to dataset
    dataset = tf.data.Dataset.from_tensor_slices(features)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()