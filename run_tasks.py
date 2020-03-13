"""
@Author Marek Szakacs
This code is part of bachelor thesis: Neural Turing Machines for modelling attention
NTM implementation was taken from https://github.com/wchen342/NeuralTuringMachine

Run this program to execute multiple training runs on a single dataset with different hyperparameters using HParams.
Training can take couple of days based on the variability of defined hyperparemeter values.

Command-line arguments provide a way to specify the dataset, number of epochs and NTM architecture properties.
If not arguments are provided, program takes default values.

Run: python run_tasks_single_parameters.py --<argument> <value> --<argument> <value> ... --<argument> <value>
Example: python run_tasks_single_parameters.py --num_epochs 10 --dataset_dir dataset/1
"""

from time import time
import tensorflow as tf
from tensorflow.python import keras
from ntm import NTMCell
import argparse
import numpy as np
import cv2
import glob
from sklearn.utils import shuffle
from tensorboard.plugins.hparams import api as hp
import os

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser()

parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=100)
parser.add_argument('--num_memory_locations', type=int, default=128)
parser.add_argument('--memory_size', type=int, default=32)
parser.add_argument('--num_read_heads', type=int, default=1)
parser.add_argument('--num_write_heads', type=int, default=1)
parser.add_argument('--conv_shift_range', type=int, default=1, help='only necessary for ntm')
parser.add_argument('--clip_value', type=int, default=20, help='Maximum absolute value of controller and outputs.')
parser.add_argument('--init_mode', type=str, default='constant', help='learned | constant | random')

parser.add_argument('--optimizer', type=str, default='Adam', help='RMSProp | Adam')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=50)
parser.add_argument('--num_train_steps', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=1)

parser.add_argument('--img_dim', type=int, default=32)
parser.add_argument('--num_bits_per_vector', type=int, default=32)
parser.add_argument('--num_vectors_per_image', type=int, default=256)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--test_dataset_size', type=int, default=10000)
parser.add_argument('--dataset_dir', type=str, default='dataset/1')

args = parser.parse_args()

HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256]))
HP_NUM_READ = hp.HParam('num_read', hp.Discrete([1, 2]))
HP_NUM_WRITE = hp.HParam('num_write', hp.Discrete([1, 2]))
HP_CONV_SHIFT = hp.HParam('conv_shift', hp.Discrete([1, 2]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.1, 0.01, 0.001]))

METRIC_ACCURACY = 'accuracy'
hparams = [HP_NUM_LAYERS, HP_NUM_UNITS, HP_NUM_READ, HP_NUM_WRITE, HP_CONV_SHIFT, HP_LEARNING_RATE]


def create_directory(name):
    try:
        os.mkdir(name)
        print("Directory " + name + " created.")
    except FileExistsError:
        print("Directory " + name + " already exists.")


def load_images(images_path, img_format):
    print("Loading images...")
    image_paths = glob.glob(images_path + "/*." + img_format)
    binary_images = []
    for path in image_paths:
        img = cv2.imread(path, 0)
        bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        binary_images.append(bin_img / 255.0)
    return np.array(binary_images)


def get_batch(iter_num, x, y):
    upper_index = (iter_num + 1) * args.batch_size
    label_batch = np.vstack(y[iter_num * args.batch_size: upper_index])
    image_batch = x[iter_num * args.batch_size: upper_index]
    return image_batch, label_batch


def shuffle_dataset(ds, lbls):
    dataset, labels = shuffle(ds, lbls, random_state=np.random.randint(50))
    return dataset, labels


class BuildModel(keras.Model):
    def __init__(self, h_params):
        super(BuildModel, self).__init__()
        self._build_model(h_params)

    def _build_model(self, h_params):

        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding="same", activation='relu')
        self.max1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding="same", activation='relu')
        self.max2 = tf.keras.layers.MaxPooling2D((2, 2))

        cell = NTMCell(h_params[HP_NUM_LAYERS], h_params[HP_NUM_UNITS], args.num_memory_locations, args.memory_size,
                       h_params[HP_NUM_READ], h_params[HP_NUM_WRITE], addressing_mode='content_and_location',
                       shift_range=h_params[HP_CONV_SHIFT], output_dim=args.num_bits_per_vector,
                       clip_value=args.clip_value, init_mode=args.init_mode)

        # output - 3D tensor with shape (batch_size, timesteps, units).   64 x 64 x 32
        self.rnn = keras.layers.RNN(
            cell=cell, return_sequences=False, return_state=False,
            stateful=False, unroll=True)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

        if args.optimizer == 'RMSProp':
            self.optimizer = tf.keras.optimizers.RMSprop(h_params[HP_LEARNING_RATE], rho=0.9, decay=0.9)
        elif args.optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(lr=h_params[HP_LEARNING_RATE])

    def call(self, inputs):
        x = tf.reshape(inputs, [-1, args.img_dim, args.img_dim, 1])  # Conv2D requires 4D tensor [batch_size, img_dim, img_dim, channels]
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2], 32])
        output_sequence = self.rnn(x)
        outputs = self.dense(output_sequence)
        return outputs


images = load_images(images_path=args.dataset_dir, img_format="png")

train_dataset = np.concatenate((images[:20000], images[30000:50000]))
test_dataset = np.concatenate((images[20000:30000], images[50000:60000]))

train_labels = np.concatenate(([[1, 0]] * 20000, [[0, 1]] * 20000))
test_labels = np.concatenate(([[1, 0]] * 10000, [[0, 1]] * 10000))
test_dataset, test_labels = shuffle_dataset(test_dataset, test_labels)


def get_apply_grad_fn():
    @tf.function
    def run_train_step(ntmModel, inputs, labels):
        # Cast data type
        inputs = tf.cast(inputs, tf.float32)
        labels = tf.cast(labels, tf.float32)

        with tf.GradientTape() as tape:
            outputs = ntmModel(inputs)
            prediction = tf.nn.softmax(outputs)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels))
        gradients = tape.gradient(loss, ntmModel.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, args.max_grad_norm)
        ntmModel.optimizer.apply_gradients(zip(gradients, ntmModel.trainable_variables))

        return loss, outputs, accuracy
    return run_train_step


@tf.function
def run_eval_step(inputs, labels, seq_len):
    # Cast data type
    inputs = tf.cast(inputs, tf.float32)
    labels = tf.cast(labels, tf.float32)
    outputs = model(inputs, seq_len)
    loss = model.loss(labels[..., tf.newaxis], outputs[..., tf.newaxis])
    loss = tf.reduce_sum(loss) / inputs.shape[0]

    return loss, outputs

create_directory("logs")

prev_time = float("-inf")
curr_time = float("-inf")
session_num = 0

for learning_rate in HP_LEARNING_RATE.domain.values:
    for num_layers in HP_NUM_LAYERS.domain.values:
        for num_read_heads in HP_NUM_READ.domain.values:
            for num_write_heads in HP_NUM_WRITE.domain.values:
                for conv_shift in HP_CONV_SHIFT.domain.values:
                    for num_units in HP_NUM_UNITS.domain.values:

                        hparams = {
                            HP_NUM_LAYERS: num_layers,
                            HP_NUM_UNITS: num_units,
                            HP_NUM_READ: num_read_heads,
                            HP_NUM_WRITE: num_write_heads,
                            HP_CONV_SHIFT: conv_shift,
                            HP_LEARNING_RATE: learning_rate
                        }

                        model = BuildModel(hparams)
                        apply_grad_fn = get_apply_grad_fn()

                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})

                        with tf.summary.create_file_writer("logs/hparam_tuning" + args.dataset_dir + "/" + run_name).as_default():
                            for i in range(args.num_epochs):
                                train_ds, train_lbls = shuffle_dataset(train_dataset, train_labels)
                                for iter in range(args.dataset_size // args.batch_size):
                                    inputs, labels = get_batch(iter, train_ds, train_lbls)
                                    train_loss, outputs, accuracy = apply_grad_fn(model, inputs, labels)
                                    train_loss = train_loss.numpy()
                                    accuracy = accuracy.numpy()
                                    if iter % 200 == 0:  # calculate and output training accuracy every 200th iteration
                                        curr_time = time()
                                        elapsed = curr_time - prev_time
                                        print(
                                            "Now at iteration %d. Elapsed time: %.5fs. Average time: %.5fs/iter" % (
                                            iter, elapsed, elapsed / 100.))
                                        prev_time = curr_time
                                        print("Epoch " + str(i))
                                        print(train_loss)
                                        tf.summary.scalar("loss", train_loss,  i * (args.dataset_size // args.batch_size) + iter, "Training loss")
                                        print(accuracy)

                            print("Training finished")
                            # Calculate accuracy
                            print("Testing Accuracy:")
                            total_accuracy = 0
                            for i in range(args.test_dataset_size // args.batch_size):
                                batch_x, batch_y = get_batch(i, test_dataset, test_labels)
                                _, _, accuracy = apply_grad_fn(model, batch_x, batch_y)
                                total_accuracy += accuracy
                            total_accuracy = total_accuracy / (args.test_dataset_size // args.batch_size)
                            print("Accuracy:", total_accuracy)

                            hp.hparams(hparams)  # record the values used in this trial
                            tf.summary.scalar(METRIC_ACCURACY, total_accuracy, step=1)
                            session_num += 1
