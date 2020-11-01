import urllib.request
import argparse
import shutil
import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import Resizing

HUB_URL = "https://tfhub.dev/tensorflow"
MODEL_NAME = "efficientnet"

_MODEL_SIZES_ = {
    "small": (os.path.join(HUB_URL, MODEL_NAME,
                           "b0", "classification", "1"), 224),
    "medium": (os.path.join(HUB_URL, MODEL_NAME,
                            "b3", "classification", "1"), 300),
    "big": (os.path.join(HUB_URL, MODEL_NAME,
                         "b5", "classification", "1"), 528)
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
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = img[tf.newaxis, ...]
            return img

        def id_image_prediction(x):
            """
            Return the label with highest logits
            """
            return self.labels[tf.argmax(x)]

        x = _read_image(x)
        rescaled_x = Resizing(self.size, self.size)(x)
        logits = self.model(rescaled_x)
        id_to_text = tf.map_fn(id_image_prediction, logits, tf.string)
        return id_to_text


def run(model_size, output_dir, del_model):
    output_dir_ = os.path.join(output_dir, MODEL_NAME)
    if not os.path.isdir(output_dir_):
        os.makedirs(output_dir_, exist_ok=True)
    model_versions = os.listdir(output_dir_)
    last_version = 0
    if len(model_versions) > 0:
        last_version = max([int(v) for v in model_versions])
    model = ImportModelHub(model_size)
    # Set incremental values of folders
    last_version += 1
    save_path = os.path.join(output_dir, MODEL_NAME, str(last_version))
    tf.saved_model.save(model, save_path)
    if del_model:
        subfolders_model = os.listdir(save_path)
        if (
            "saved_model.pb" in subfolders_model
            and set(["assets", "variables"]) <= set(subfolders_model)
           ):
            print("Model '{}', version '{}'".format(MODEL_NAME, last_version),
                  "has been saved correctly. Deleting folders...")
        else:
            print("An error has ocurred while saving,",
                  "model '{}', version '{}".format(MODEL_NAME, last_version))
        shutil.rmtree(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',
                        choices=list(_MODEL_SIZES_.keys()),
                        help="Choose the size of models to be saved")
    parser.add_argument('--output_dir',
                        type=str,
                        default="./models",
                        help="Output path where models are saved")
    parser.add_argument('-d',
                        action="store_true",
                        help="Check for deleting a models that has been created correctly")
    args = parser.parse_args()

    # Run the script with proper arguments
    run(args.size, args.output_dir, args.d)
