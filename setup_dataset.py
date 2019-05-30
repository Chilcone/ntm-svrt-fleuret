import tensorflow as tf
import glob
import numpy as np
from PIL import Image
#from skimage.util import view_as_windows
import os

patch_size = 4
dataset_dir_path = "dataset"

class SetupDataset:
    def __init__(self, patch_size, dataset_dir_path):
        self.patch_size = patch_size
        self.dataset_dir_path = dataset_dir_path

    def load_images(self, images_path):
        image_names = glob.glob(images_path + "/*.png")
        images = np.array([np.array(Image.open(img).convert("L")) for img in image_names])/255.0
        return images

    '''def extract_patches(self, category):
        patches = view_as_windows(category, (patch_size, patch_size), patch_size)  # non-overlapping patches
        return patches'''

    def get_cnn_feature_vectors(self, dataset):
        extracted_features = []
        for i in range(len(dataset)):
            if (i % 1000 == 0):
                print("Processing image ", i + 1)
            # Input Layer
            input_layer = tf.reshape(dataset[i], [1, 32, 32, 1])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=16,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            feature_vector = tf.reshape(pool2, [64, 32])
            extracted_features.append(feature_vector)
        return tf.stack(extracted_features)

    def create_directory(self, name):
        try:
            os.mkdir(name)
            print("Directory " + name + " created.")
        except FileExistsError:
            print("Directory " + name + " already exists.")

    def process_and_save_dataset(self, dirName):
        image_directories = os.listdir(self.dataset_dir_path)
        self.create_directory(dirName)

        with tf.Session() as sess:
            for directory in image_directories[:1]:
                images = self.load_images(self.dataset_dir_path + "/" + directory)
                for i in range(len(images) // 10000):
                    processed_images = self.get_cnn_feature_vectors(images[i * 10000:(i+1)*10000])

                    print("Initializing TensorFlow variable...")
                    sess.run(tf.global_variables_initializer())

                    self.create_directory(dirName + "/" + directory)
                    save_path =  dirName + "/" + directory + "/part" + str(i)
                    print("Saving processed images representation...")
                    np.save(save_path, processed_images.eval())

dataset = SetupDataset(patch_size, dataset_dir_path)
dataset.process_and_save_dataset("processed_dataset")


