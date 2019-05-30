import numpy as np
from sklearn.utils import shuffle

class FleuretTaskData:
    def __init__(self):
        # load already preprocessed images from setup_dataset.py, each part contains 10000 examples
        class1_a = np.load("processed_dataset/1/part0.npy")
        class1_b = np.load("processed_dataset/1/part1.npy")
        class2_a = np.load("processed_dataset/1/part3.npy")
        class2_b = np.load("processed_dataset/1/part4.npy")

        self.dataset = np.concatenate((class1_a, class1_b, class2_a, class2_b))
        self.labels = np.concatenate(([0] * 20000, [1] * 20000))

    def shuffle_dataset(self):
        print("Shuffling dataset...")
        self.dataset, self.labels = shuffle(self.dataset, self.labels, random_state=0)
        return self.dataset, self.labels

    def get_batch(self, iter_num, batch_size):
        upper_index = (iter_num+1) * batch_size
        return self.dataset[iter_num * batch_size: upper_index], np.vstack(self.labels[iter_num * batch_size: upper_index])
