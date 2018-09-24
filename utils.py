import glob
import tensorflow as tf

def read_all_images(path,x,y):
    files = glob.glob (path)
    data = []
    for myFile in files:
        # print myFile
        image = tf.image.decode_jpeg(myFile,channels=3)
        resized_image = tf.image.resize_images(image, [x, y])
        data.append(resized_image)
    return tf.stack(data)


# def read_all_images1(path,x,y):
#     filename_queue = tf.train.string_input_producer(
#             tf.train.match_filenames_once(path))
#     image_reader = tf.WholeFileReader()
#     _, image_file = image_reader.read(filename_queue)
#     image = tf.image.decode_jpeg(image_file)
#     resized_image = tf.image.resize_images(image, [x, y])
#     return resized_image

def train_input_fn(features, batch_size):
    """ Input function for training """
    #convert input to dataset
    dataset = tf.data.Dataset.from_tensor_slices(features)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()