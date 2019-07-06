from os.path import abspath, dirname, join

import h5py
import matplotlib.pyplot as plt
import numpy as np

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
        for key, val in self.__dict__.items():
            group.create_dataset(key, data=val)

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
    def __init__(self, env_name, timestamp, log_dir, *args):
        """
        Creates a TemporalLogger object. If the folder structure is nonexistent, it will also be created
        :param *args:
        :param env_name: name of the environment
        :param timestamp: timestamp as a string
        :param log_dir: logging directory, if it is None, then logging will be at the same hierarchy level as src/
        """
        super().__init__()
        self.timestamp = timestamp

        # file structure
        self.base_dir = join(dirname(dirname(abspath(__file__))), "log") if log_dir is None else log_dir
        self.data_dir = join(self.base_dir, env_name)
        make_dir(self.base_dir)
        make_dir(self.data_dir)

        # data
        for data in args:
            self.__dict__[data] = LogData()

    def log(self, **kwargs):
        """
        Function for storing the new values of the given attribute
        :param **kwargs:
        :return:
        """
        for key, value in kwargs.items():
            self.__dict__[key].log(value)

    def save(self, *args):
        """
        Saves the temporal statistics into a .hdf5 file
        :param **kwargs:
        :return:
        """
        with h5py.File(join(self.data_dir, 'time_log_' + self.timestamp + '.hdf5'), 'w') as f:
            for arg in args:
                self.__dict__[arg].save(f.create_group(arg))

    def load(self, filename):
        """
        Loads the temporal statistics and fills the attributes of the class
        :param filename: name of the .hdf5 file to load
        :return:
        """
        if not filename.endswith('.hdf5'):
            filename = filename + '.hdf5'

        with h5py.File(join(self.data_dir, filename), 'r') as f:
            for key, value in self.__dict__.items():
                if isinstance(value, LogData):
                    value.load(f[key])

    def plot_mean_min_max(self, *args):
        for arg in args:
            # breakpoint()
            if arg in self.__dict__.keys():  # and isinstance(self.__dict__[arg], LogData):
                self.__dict__[arg].plot_mean_min_max(arg)
        plt.title("Mean and min-max statistics")
        self._print_plot_details()

    def plot_mean_std(self, *args):
        for arg in args:
            if arg in self.__dict__.keys():
                self.__dict__[arg].plot_mean_std(arg)

        plt.title("Mean and standard deviation statistics")
        self._print_plot_details()

    def _print_plot_details(self):
        plt.xlabel("Rollout")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
