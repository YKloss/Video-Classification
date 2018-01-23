"""
Class for managing our data.
"""
import numpy as np
import glob
import os.path
import operator
import threading
from keras.utils import to_categorical
import pandas as pd
from image_processor import process_image


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""

    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))

    return gen


class DataSet:
    def __init__(self, seq_length=40, class_limit=None, image_shape=(299, 299, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences', 'inceptionv3')
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        return pd.read_csv(os.path.join('data', 'data_files.csv'),
                           names=["train_test", "class_name", "file_name", "frame_count"]
                           )

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        return self.data[(self.seq_length <= self.data["frame_count"]) & (self.data["frame_count"] <= self.max_frames)]

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""

        return list(self.data["class_name"].unique())

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = self.data[(self.data["train_test"] == "train")]
        test = self.data[(self.data["train_test"] == "test")]

        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for index, row in data.iterrows():

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    raise Exception("Can't find sequence. Did you generate them?")

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while True:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = data.sample()

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y.append(self.get_class_one_hot(sample.iloc[0]["class_name"]))

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        try:
            filename = sample.iloc[0]['file_name']
        except:
            filename = sample['file_name']
        path = os.path.join(self.sequence_path, str(filename) + '-' + str(self.seq_length) + '-' + str(data_type) + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        try:
            path = os.path.join('data', sample.iloc[0]["train_test"], sample.iloc[0]["class_name"])
            filename = sample.iloc[0]["file_name"]
        except:
            path = os.path.join('data', sample["train_test"], sample["class_name"])
            filename = sample["file_name"]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        # parts = filename.split(os.path.sep)
        head, filename = os.path.split(filename)
        return filename.replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the original list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.data.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
