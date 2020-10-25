import urllib.request
import argparse
import os

import tensorflow as tf
import tensorflow_hub as hub


_MODEL_SIZES_ = {
    "small": ("https://tfhub.dev/tensorflow/efficientnet/b0/classification/1", 224), 
    "medium": ("https://tfhub.dev/tensorflow/efficientnet/b3/classification/1", 300),
    "big": ("https://tfhub.dev/tensorflow/efficientnet/b3/classification/1", 528)
    }

FILE_LABELS = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

class ImportModelHub(tf.keras.Model):
    """
    Class to load a pretrained model from Tensorflow Hub.
    The user can choose the size of the model based on the
    number of parameters.
    """
    def __init__(self, size):
        super(ImportModelHub, self).__init__()
        url_handle, img_size = _MODEL_SIZES_[size]
        self.size = img_size
        self.model = hub.load(url_handle)
        self.labels = self.read_labels(FILE_LABELS, "labels.txt")

    def read_labels(self, url, output_file):
        if not os.path.exists(output_file):
            urllib.request.urlretrieve(FILE_LABELS, output_file)
        labels_str = tf.io.read_file(output_file)
        return tf.strings.split(labels_str, sep='\n')

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)])
    def __call__(self, x):
        def _read_image(img_file):
            img_bytes = tf.reshape(img_file, [])
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis,...]
            return img

        x = _read_image(x)
        rescaled_x = tf.keras.layers.experimental.preprocessing.Resizing(self.size, self.size)(x)
        logits = self.model(rescaled_x)
        id_image_prediction = lambda x: self.labels[tf.argmax(x)]
        id_to_text = tf.map_fn(id_image_prediction, logits, tf.string)
        return id_to_text

            
def run(model_size, output_dir):
    output_dir_ = os.path.join(output_dir, "efficientnet") 
    if not os.path.isdir(output_dir_):
        os.makedirs(output_dir_,exist_ok=True)
    model_versions = os.listdir(output_dir_)
    last_version = max([int(v) for v in model_versions]) if len(model_versions) > 0 else 0 
    model = ImportModelHub(model_size)
    # Set incremental values of folders
    last_version += 1
    tf.saved_model.save(model, os.path.join(output_dir, "efficientnet", str(last_version)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=list(_MODEL_SIZES_.keys()), help="Choose the size of models to be saved")
    parser.add_argument('--output_dir', type=str, default="./models", help="Output path where models are saved")
    args = parser.parse_args()

    # Run the script with proper arguments
    run(args.size, args.output_dir)
