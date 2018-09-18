import glob
import tensorflow as tf

def read_all_images(path,x,y):
    files = glob.glob (path)
    data = []
    for myFile in files:
        image = tf.image.decode_jpeg(myFile,channels=3)
        resized_image = tf.image.resize_images(image, [x, y])
        data.append(resized_image)
    return tf.stack(data)


def train_input_fn(features, batch_size):
    """ Input function for training """
    #convert input to dataset
    dataset = tf.data.Dataset.from_tensor_slices(features)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()