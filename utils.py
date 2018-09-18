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