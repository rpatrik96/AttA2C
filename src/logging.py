import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import abspath, dirname, join

from utils import make_dir

class LogData(object):
    def __init__(self):
        self.mean = []
        self.std = []
        self.min = []
        self.max = []

    def log(self, sample):
        """

        :param sample: data for logging specified as a numpy.array
        :return:
        """
        self.mean.append(sample.mean())
        self.std.append(sample.std())
        self.min.append(sample.min())
        self.max.append(sample.max())

    def save(self, group):
        """

        :param group: the reference to the group level hierarchy of a .hdf5 file to save the data
        :return:
        """
        group.create_dataset("mean", data=self.mean)
        group.create_dataset("std", data=self.std)
        group.create_dataset("min", data=self.min)
        group.create_dataset("max", data=self.max)

    def load(self, group):
        """
        :param group: the reference to the group level hierarchy of a .hdf5 file to load
        :return:
        """
        # read in parameters
        # [()] is needed to read in the whole array if you don't do that,
        #  it doesn't read the whole data but instead gives you lazy access to sub-parts
        #  (very useful when the array is huge but you only need a small part of it).
        # https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
        self.mean = group["mean"][()]
        self.std = group["std"][()]
        self.min = group["min"][()]
        self.max = group["max"][()]

    def plot_mean_min_max(self, label):
        plt.fill_between(range(len(self.mean)), self.max, self.min, alpha=.5)
        plt.plot(self.mean, label=label)

    def plot_mean_std(self, label):
        mean = np.array(self.mean)
        plt.fill_between(range(len(self.mean)), mean + self.std, mean - self.std, alpha=.5)
        plt.plot(self.mean, label=label)


class TemporalLogger(object):
    def __init__(self, env_name, timestamp):
        """
        Creates a TemporalLogger object. If the folder structure is nonexistent, it will also be created
        :param env_name: name of the environment
        :param timestamp: timestamp as a string
        """
        super().__init__()
        self.timestamp = timestamp

        # file structure
        self.base_dir = join(dirname(dirname(abspath(__file__))), "log")
        self.data_dir = join(self.base_dir, env_name)
        make_dir(self.base_dir)
        make_dir(self.data_dir)

        # data
        self.rewards = LogData()
        self.features = LogData()

    def log(self, reward, feature):
        """
        Function for storing the new values of the given attribute
        :param reward: reward as numpy.array
        :param feature: feature as numpy.array
        :return:
        """
        self.rewards.log(reward)
        self.features.log(feature)

    def save(self):
        """
        Saves the temporal statistics into a .hdf5 file
        :return:
        """
        with h5py.File(join(self.data_dir, "time_log_" + self.timestamp + '.hdf5'), 'w') as f:
            rewards = f.create_group("rewards")
            features = f.create_group("features")

            self.rewards.save(rewards)
            self.features.save(features)

    def load(self, filename):
        """
        Loads the temporal statistics and fills the attributes of the class
        :param filename: name of the .hdf5 file to load
        :return:
        """
        with h5py.File(join(self.data_dir, filename + '.hdf5'), 'r') as f:
            print(f['rewards'])
            self.rewards.load(f['rewards'])
            self.features.load(f['features'])

    def plot_mean_min_max(self):
        self.rewards.plot_mean_min_max("rewards")
        self.features.plot_mean_min_max("features")
        plt.title("Mean and min-max statistics")
        self._print_plot_details()

    def plot_mean_std(self):
        self.rewards.plot_mean_std("rewards")
        self.features.plot_mean_std("features")
        plt.title("Mean and standard deviation statistics")
        self._print_plot_details()

    def _print_plot_details(self):
        plt.xlabel("Rollout")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

